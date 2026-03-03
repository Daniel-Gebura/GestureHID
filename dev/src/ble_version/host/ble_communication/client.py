################################################################
# ble_communication/client.py
#
# Description:
#  A reliable, professional BLE "messenger" client for your nRF52840 UART HID proxy.
#
#   Key design points:
#   - Async-first (BLE is inherently async), but easy to wrap if needed.
#   - Clean API: you call methods like `keyboard_write()` or `mouse_move()`.
#   - Retries on write (Windows BLE can cancel writes during pairing/settling).
#   - Optional ack verification for each command.
#   - Explicit connect/auth lifecycle.
#
################################################################

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional, Callable

from bleak import BleakClient, BleakScanner

from .constants import NUS_RX_CHAR_UUID, NUS_TX_CHAR_UUID, DEFAULT_MAX_JSON_BYTES
from .exceptions import (
    DeviceNotFoundError,
    ConnectionError,
    AuthenticationError,
    AckTimeoutError,
    DeviceErrorResponse,
)
from .models import DeviceTarget, Ack
from .utils import json_dumps_compact, ensure_trailing_newline
from . import protocol


@dataclass
class BleHidProxyConfig:
    """
    Configuration knobs for reliability vs latency.
    """
    scan_timeout_s: float = 10.0
    connect_timeout_s: float = 15.0
    post_connect_settle_s: float = 0.5

    # Write retry policy
    write_retries: int = 5
    write_retry_delay_s: float = 0.5

    # Ack policy
    default_ack_timeout_s: float = 1.0

    # Payload cap for sanity
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES


class BleHidProxyClient:
    """
    A high-level, reliable BLE client for your HID proxy device.

    Typical usage:

        client = BleHidProxyClient(DeviceTarget(name="HID-Proxy-Control"), token="CHANGE_ME")
        await client.connect()
        await client.keyboard_write("Hello\\n")
        await client.mouse_move(dx=20)
        await client.disconnect()

    Notes:
        - This client expects the device to respond with JSON acks on the TX characteristic.
        - Your device already sends {"ok":true} or {"ok":false,"error":"..."}.
    """

    def __init__(
        self,
        target: DeviceTarget,
        token: str,
        config: Optional[BleHidProxyConfig] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._target = target
        self._token = token
        self._cfg = config or BleHidProxyConfig()
        self._log = on_log or (lambda _msg: None)

        self._client: Optional[BleakClient] = None
        self._notify_queue: asyncio.Queue[str] = asyncio.Queue()
        self._notify_buffer: str = ""
        self._connected: bool = False
        self._authed: bool = False

    # --------------------------
    # Public lifecycle
    # --------------------------

    async def connect(self) -> None:
        """
        Discover, connect, start notifications, and authenticate.
        """
        if self._connected:
            return

        device = await self._discover_device()
        self._client = BleakClient(device)

        try:
            await asyncio.wait_for(self._client.connect(), timeout=self._cfg.connect_timeout_s)
        except Exception as e:
            raise ConnectionError(f"BLE connect failed: {e!r}") from e

        self._connected = True
        self._log("Connected.")

        # Notifications are required for reading acks from the device.
        try:
            await self._client.start_notify(NUS_TX_CHAR_UUID, self._on_notify)
        except Exception as e:
            await self._safe_disconnect()
            raise ConnectionError(f"Failed to start notifications: {e!r}") from e

        # Let Windows (and some BLE stacks) settle.
        await asyncio.sleep(self._cfg.post_connect_settle_s)

        # Best-effort OS pairing (often already done in OS settings)
        try:
            await self._client.pair()
        except Exception:
            pass

        # Authenticate
        await self.authenticate()

    async def disconnect(self) -> None:
        """
        Stop notifications and disconnect.
        """
        await self._safe_disconnect()

    async def authenticate(self) -> None:
        """
        Send {"type":"auth","token":...} and require ok=true.
        """
        if not self._connected or self._client is None:
            raise ConnectionError("Not connected")

        # Send auth (with retries) and wait for ack
        ack = await self._send_and_maybe_wait_ack(
            protocol.cmd_auth(self._token),
            require_ack=True,
            ack_timeout_s=3.0,
        )

        if not ack.ok:
            raise AuthenticationError(f"Auth rejected: {ack.error}")

        self._authed = True
        self._log("Authenticated.")

    # --------------------------
    # Public command API
    # --------------------------

    async def keyboard_write(self, text: str, require_ack: bool = True, ack_timeout_s: Optional[float] = None) -> Ack:
        return await self._send_and_maybe_wait_ack(
            protocol.cmd_keyboard_write(text),
            require_ack=require_ack,
            ack_timeout_s=ack_timeout_s,
        )

    async def keyboard_press(self, keys: list[str], require_ack: bool = True, ack_timeout_s: Optional[float] = None) -> Ack:
        return await self._send_and_maybe_wait_ack(
            protocol.cmd_keyboard_press(keys),
            require_ack=require_ack,
            ack_timeout_s=ack_timeout_s,
        )

    async def keyboard_release(self, keys: list[str], require_ack: bool = True, ack_timeout_s: Optional[float] = None) -> Ack:
        return await self._send_and_maybe_wait_ack(
            protocol.cmd_keyboard_release(keys),
            require_ack=require_ack,
            ack_timeout_s=ack_timeout_s,
        )

    async def keyboard_release_all(self, require_ack: bool = True, ack_timeout_s: Optional[float] = None) -> Ack:
        return await self._send_and_maybe_wait_ack(
            protocol.cmd_keyboard_release_all(),
            require_ack=require_ack,
            ack_timeout_s=ack_timeout_s,
        )

    async def mouse_move(
        self,
        dx: int = 0,
        dy: int = 0,
        wheel: int = 0,
        require_ack: bool = False,
        ack_timeout_s: Optional[float] = None,
    ) -> Ack:
        return await self._send_and_maybe_wait_ack(
            protocol.cmd_mouse_move(dx=dx, dy=dy, wheel=wheel),
            require_ack=require_ack,
            ack_timeout_s=ack_timeout_s,
        )

    async def mouse_click(self, button: str = "LMB", require_ack: bool = True, ack_timeout_s: Optional[float] = None) -> Ack:
        return await self._send_and_maybe_wait_ack(
            protocol.cmd_mouse_click(button),
            require_ack=require_ack,
            ack_timeout_s=ack_timeout_s,
        )

    async def mouse_press(self, button: str = "LMB", require_ack: bool = True, ack_timeout_s: Optional[float] = None) -> Ack:
        return await self._send_and_maybe_wait_ack(
            protocol.cmd_mouse_press(button),
            require_ack=require_ack,
            ack_timeout_s=ack_timeout_s,
        )

    async def mouse_release(self, button: str = "LMB", require_ack: bool = True, ack_timeout_s: Optional[float] = None) -> Ack:
        return await self._send_and_maybe_wait_ack(
            protocol.cmd_mouse_release(button),
            require_ack=require_ack,
            ack_timeout_s=ack_timeout_s,
        )

    # --------------------------
    # Internals
    # --------------------------

    async def _discover_device(self):
        """
        Discover device by address (if provided) or name (otherwise).
        """
        if self._target.address:
            # Bleak can take an address directly on some platforms,
            # but discovery-by-address behavior varies. We still scan and match.
            self._log(f"Scanning for address={self._target.address} ...")
        else:
            self._log(f"Scanning for name={self._target.name!r} ...")

        devices = await BleakScanner.discover(timeout=self._cfg.scan_timeout_s)

        for d in devices:
            if self._target.address and d.address.lower() == self._target.address.lower():
                return d
            if not self._target.address and d.name == self._target.name:
                return d

        raise DeviceNotFoundError(f"Could not find device (name={self._target.name!r}, address={self._target.address!r})")

    def _on_notify(self, _sender: int, data: bytearray) -> None:
        """
        Notification callback from Bleak. Push chunks into an asyncio queue.
        """
        try:
            text = bytes(data).decode("utf-8", errors="replace")
        except Exception:
            text = ""
        try:
            self._notify_queue.put_nowait(text)
        except Exception:
            pass

    async def _safe_disconnect(self) -> None:
        """
        Best-effort cleanup.
        """
        if self._client is None:
            self._connected = False
            self._authed = False
            return

        try:
            try:
                await self._client.stop_notify(NUS_TX_CHAR_UUID)
            except Exception:
                pass

            await self._client.disconnect()
        except Exception:
            pass
        finally:
            self._client = None
            self._connected = False
            self._authed = False
            self._notify_buffer = ""
            # Drain queue
            try:
                while not self._notify_queue.empty():
                    _ = self._notify_queue.get_nowait()
            except Exception:
                pass
            self._log("Disconnected.")

    async def _write_line(self, payload_dict: dict) -> None:
        """
        Serialize + send one JSON line with retries.
        """
        if not self._connected or self._client is None:
            raise ConnectionError("Not connected")

        s = json_dumps_compact(payload_dict)
        s = ensure_trailing_newline(s)
        b = s.encode("utf-8")

        if len(b) > self._cfg.max_json_bytes:
            raise ValueError(f"Command payload too large ({len(b)} bytes)")

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._cfg.write_retries + 1):
            try:
                await self._client.write_gatt_char(NUS_RX_CHAR_UUID, b, response=False)
                return
            except Exception as e:
                last_exc = e
                self._log(f"Write failed (attempt {attempt}/{self._cfg.write_retries}): {e!r}")
                await asyncio.sleep(self._cfg.write_retry_delay_s)

        raise ConnectionError(f"GATT write failed after retries: {last_exc!r}")

    async def _read_ack(self, timeout_s: float) -> Ack:
        """
        Wait for one JSON line from device notifications.
        """
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            # Pull new chunk
            remaining = max(0.0, deadline - time.time())
            try:
                chunk = await asyncio.wait_for(self._notify_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break

            self._notify_buffer += chunk

            # Parse complete lines
            while "\n" in self._notify_buffer:
                line, self._notify_buffer = self._notify_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception:
                    # Ignore noise / partials / non-json
                    continue

                ok = bool(obj.get("ok", False))
                err = obj.get("error", None)
                return Ack(ok=ok, error=err, raw=obj)

        raise AckTimeoutError(f"Ack timeout after {timeout_s}s")

    async def _send_and_maybe_wait_ack(
        self,
        payload_dict: dict,
        require_ack: bool,
        ack_timeout_s: Optional[float],
    ) -> Ack:
        """
        Write a command and optionally wait for the device to ack it.
        """
        # Guard: do not allow non-auth commands before auth
        if payload_dict.get("type") != "auth" and not self._authed:
            raise AuthenticationError("Not authenticated")

        await self._write_line(payload_dict)

        if not require_ack:
            return Ack(ok=True, error=None, raw={"ok": True, "note": "ack_skipped"})

        t = ack_timeout_s if ack_timeout_s is not None else self._cfg.default_ack_timeout_s
        ack = await self._read_ack(timeout_s=t)

        if not ack.ok:
            raise DeviceErrorResponse(ack.error or "device_error", payload=ack.raw)

        return ack