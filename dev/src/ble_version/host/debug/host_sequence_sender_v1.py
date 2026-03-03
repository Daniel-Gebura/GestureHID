#!/usr/bin/env python3
################################################################
# host_sequence_sender_v1.py
#
# Description:
#   Windows/PC-side BLE "central" that connects to your nRF52840
#   (Nordic UART Service / UARTService), authenticates, then sends a
#   deterministic sequence of keyboard + mouse commands.
#
#   This is NOT a live passthrough. It's a scripted "messenger"
#   for testing the full pipeline end-to-end:
#       PC -> BLE UART JSON -> nRF -> USB HID -> Target PC
#
# Requirements:
#   pip install bleak
#
# Usage:
#   python host_sequence_sender.py --name HID-Proxy-Control --token CHANGE_ME
#
################################################################

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

from bleak import BleakClient, BleakScanner


# ----------------------------------------------------------------------
# NUS (Nordic UART Service) UUIDs (standard)
# ----------------------------------------------------------------------

NUS_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # Write to peripheral
NUS_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # Notifications from peripheral


# ----------------------------------------------------------------------
# BLE UART Link
# ----------------------------------------------------------------------

@dataclass
class BleUartLink:
    client: BleakClient
    rx_uuid: str = NUS_RX_CHAR_UUID
    tx_uuid: str = NUS_TX_CHAR_UUID
    _notify_queue: asyncio.Queue = None

    def __post_init__(self) -> None:
        self._notify_queue = asyncio.Queue()

    def _on_notify(self, _sender: int, data: bytearray) -> None:
        try:
            text = bytes(data).decode("utf-8", errors="replace")
        except Exception:
            text = ""
        self._notify_queue.put_nowait(text)

    async def start_notifications(self) -> None:
        await self.client.start_notify(self.tx_uuid, self._on_notify)

    async def stop_notifications(self) -> None:
        try:
            await self.client.stop_notify(self.tx_uuid)
        except Exception:
            pass

    async def write_line(self, obj: dict) -> None:
        payload = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        await self.client.write_gatt_char(self.rx_uuid, payload, response=False)

    async def read_json_response(self, timeout_s: float = 2.0) -> Optional[dict]:
        deadline = time.time() + timeout_s
        buf = ""

        while time.time() < deadline:
            try:
                chunk = await asyncio.wait_for(self._notify_queue.get(), timeout=deadline - time.time())
            except asyncio.TimeoutError:
                break

            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except Exception:
                    continue

        return None


# ----------------------------------------------------------------------
# Discovery / Connect
# ----------------------------------------------------------------------

async def find_device_by_name(name: str, scan_timeout_s: float) -> Optional[object]:
    devices = await BleakScanner.discover(timeout=scan_timeout_s)
    for d in devices:
        if d.name == name:
            return d
    return None


async def connect_and_auth(name: str, token: str, scan_timeout_s: float) -> BleUartLink:
    dev = await find_device_by_name(name, scan_timeout_s)
    if dev is None:
        raise RuntimeError(f"Device {name!r} not found. Is it advertising?")

    client = BleakClient(dev)
    await client.connect()

    link = BleUartLink(client=client)
    await link.start_notifications()

    # Let the BLE link settle (Windows can be finicky immediately after connect)
    await asyncio.sleep(0.5)

    # Best-effort pairing. You already paired in Windows Settings; this is just extra.
    try:
        await client.pair()
    except Exception:
        pass

    # Auth with retries (Windows can cancel writes during pairing/settling)
    last_exc = None
    for attempt in range(1, 6):
        try:
            await link.write_line({"type": "auth", "token": token})
            break
        except Exception as e:
            last_exc = e
            await asyncio.sleep(0.5)
    else:
        raise RuntimeError(f"Failed to send auth after retries: {last_exc!r}")

    resp = await link.read_json_response(timeout_s=3.0)
    if not resp or not resp.get("ok", False):
        raise RuntimeError(f"Auth failed or no ack. Response={resp}")

    return link


# ----------------------------------------------------------------------
# High-level "messenger" API (what your top-level program would call)
# ----------------------------------------------------------------------

async def send_keyboard_press(link: BleUartLink, keys: list, ack: bool = True) -> None:
    await link.write_line({"type": "keyboard", "action": "press", "keys": keys})
    if ack:
        r = await link.read_json_response(timeout_s=1.0)
        print("[ACK]", r)


async def send_keyboard_release(link: BleUartLink, keys: list, ack: bool = True) -> None:
    await link.write_line({"type": "keyboard", "action": "release", "keys": keys})
    if ack:
        r = await link.read_json_response(timeout_s=1.0)
        print("[ACK]", r)


async def send_keyboard_release_all(link: BleUartLink, ack: bool = True) -> None:
    await link.write_line({"type": "keyboard", "action": "release_all"})
    if ack:
        r = await link.read_json_response(timeout_s=1.0)
        print("[ACK]", r)


async def send_keyboard_write(link: BleUartLink, text: str, ack: bool = True) -> None:
    await link.write_line({"type": "keyboard", "action": "write", "text": text})
    if ack:
        r = await link.read_json_response(timeout_s=1.0)
        print("[ACK]", r)


async def send_mouse_move(link: BleUartLink, dx: int = 0, dy: int = 0, wheel: int = 0, ack: bool = True) -> None:
    await link.write_line({"type": "mouse", "action": "move", "dx": dx, "dy": dy, "wheel": wheel})
    if ack:
        r = await link.read_json_response(timeout_s=1.0)
        print("[ACK]", r)


async def send_mouse_click(link: BleUartLink, button: str = "LMB", ack: bool = True) -> None:
    await link.write_line({"type": "mouse", "action": "click", "button": button})
    if ack:
        r = await link.read_json_response(timeout_s=1.0)
        print("[ACK]", r)


# ----------------------------------------------------------------------
# Demo Sequence
# ----------------------------------------------------------------------

async def run_demo_sequence(link: BleUartLink) -> None:
    """
    End-to-end test sequence.

    IMPORTANT:
        Put cursor focus on a text field on the TARGET PC (the one the nRF is plugged into)
        before you run this sequence, or you'll "type" into nowhere.
    """
    print("[*] Starting demo sequence in 2 seconds. Focus a text box on the TARGET PC now.")
    await asyncio.sleep(2.0)

    # 1) Type a sentence (your device supports keyboard.write)
    print("[*] Typing test message...")
    await send_keyboard_write(link, "Hello from BLE HID proxy!\n", ack=True)

    # 2) Ctrl+A then Backspace (clear text)
    print("[*] Sending CTRL+A then BACKSPACE...")
    await send_keyboard_press(link, ["CTRL"], ack=True)
    await send_keyboard_press(link, ["a"], ack=True)
    await send_keyboard_release(link, ["a"], ack=True)
    await send_keyboard_release(link, ["CTRL"], ack=True)
    await asyncio.sleep(0.1)
    await send_keyboard_press(link, ["BACKSPACE"], ack=True)
    await send_keyboard_release(link, ["BACKSPACE"], ack=True)

    await asyncio.sleep(0.3)

    # 3) Type again
    print("[*] Typing second message...")
    await send_keyboard_write(link, "Second line after clear.\n", ack=True)

    # 4) Mouse jiggle pattern
    print("[*] Mouse movement pattern...")
    for _ in range(10):
        await send_mouse_move(link, dx=20, dy=0, wheel=0, ack=False)
        await asyncio.sleep(0.02)
    for _ in range(10):
        await send_mouse_move(link, dx=-20, dy=0, wheel=0, ack=False)
        await asyncio.sleep(0.02)

    # 5) Left click
    print("[*] Mouse click...")
    await send_mouse_click(link, button="LMB", ack=True)

    # 6) Wheel scroll
    print("[*] Wheel scroll...")
    for _ in range(5):
        await send_mouse_move(link, dx=0, dy=0, wheel=-1, ack=False)
        await asyncio.sleep(0.05)

    # Always release everything at the end
    print("[*] Releasing all keys (safety)...")
    await send_keyboard_release_all(link, ack=True)

    print("[+] Demo sequence complete.")


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(description="BLE UART HID proxy: scripted sequence sender")
    parser.add_argument("--name", default="HID-Proxy-Control", help="BLE advertising name")
    parser.add_argument("--token", required=True, help="Auth token (must match AUTH_TOKEN in device code)")
    parser.add_argument("--scan-timeout", type=float, default=10.0, help="BLE scan timeout (seconds)")
    args = parser.parse_args()

    print(f"[+] Scanning for BLE device name={args.name!r} ...")
    link = await connect_and_auth(args.name, args.token, scan_timeout_s=args.scan_timeout)

    print("[+] Connected + authenticated.")
    print("[*] Running scripted demo sequence...")

    try:
        await run_demo_sequence(link)
    finally:
        try:
            await link.stop_notifications()
        except Exception:
            pass
        try:
            await link.client.disconnect()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))