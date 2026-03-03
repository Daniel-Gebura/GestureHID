################################################################
# BLEHIDController.py
#
# Description:
#   Host-side HID Controller that sends keyboard/mouse commands to your
#   Adafruit nRF52840 BLE HID proxy over BLE (Nordic UART + JSON protocol).
#
#   This class intentionally mirrors the same public API as HIDController.py:
#       - move_mouse(x, y, buttons=0x00)
#       - press_mouse(button)
#       - release_mouse()
#       - press_key(key_name, modifier=0x00)
#       - release_keys()
#       - tap_key(key_name, modifier=0x00, delay=0.1)
#
#   So your existing FSM logic can remain unchanged and simply swap output backends.
#
# Author: Daniel Gebura (integration) / ChatGPT (adapter)
################################################################

import asyncio
import threading
import time

# Keep USB HID constants identical so existing gesture maps keep working
from HIDController import MOUSE_LEFT, MOUSE_RIGHT, MOUSE_MIDDLE

from ble_communication import BleHidProxyClient, BleHidProxyConfig, DeviceTarget


class BLEHIDController:
    """
    Abstraction class to control HID mouse and keyboard events via BLE to nRF52840.

    This implementation is designed for synchronous callers (your FSMs) but internally
    uses async BLE operations. We solve that by running an asyncio loop in a dedicated
    background thread and scheduling BLE calls onto it.
    """

    def __init__(
        self,
        device_name="HID-Proxy-Control",
        auth_token="CHANGE_ME",
        scan_timeout_s=10.0,
        write_retries=5,
        write_retry_delay_s=0.5,
        default_ack_timeout_s=1.0,
    ):
        """
        Initialize BLEHIDController and prepare background BLE event loop.

        Args:
            device_name (str): BLE advertising name of the nRF device.
            auth_token (str): Auth token that must match AUTH_TOKEN in device code.
            scan_timeout_s (float): BLE scan timeout.
            write_retries (int): Retries for BLE GATT writes (Windows BLE is flaky).
            write_retry_delay_s (float): Delay between retries.
            default_ack_timeout_s (float): Default timeout for device acknowledgements.
        """
        self.device_name = device_name
        self.auth_token = auth_token

        # Build BLE client config
        self._cfg = BleHidProxyConfig(
            scan_timeout_s=scan_timeout_s,
            write_retries=write_retries,
            write_retry_delay_s=write_retry_delay_s,
            default_ack_timeout_s=default_ack_timeout_s,
        )

        # Background asyncio loop state
        self._loop = None
        self._thread = None
        self._loop_ready = threading.Event()

        # The underlying BLE client (async)
        self._client = None

        # Track pressed mouse buttons so we can release everything cleanly
        self._pressed_buttons_mask = 0x00

        # Simple lock to keep FSM calls from interleaving state updates
        self._lock = threading.Lock()

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------

    def connect(self):
        """
        Start background event loop thread, connect BLE, and authenticate.

        This should be called once at startup (before the FSM begins emitting commands).
        """
        self._start_loop_thread()

        # Create the async BLE client on the loop thread
        target = DeviceTarget(name=self.device_name)
        self._client = BleHidProxyClient(
            target=target,
            token=self.auth_token,
            config=self._cfg,
            on_log=self._log,   # Optional: prints BLE lifecycle logs
        )

        # Connect + auth
        self._run_coro_blocking(self._client.connect())

    def close(self):
        """
        Disconnect BLE and stop background loop thread.
        """
        # Best-effort release to avoid stuck state on target PC
        try:
            self.release_keys()
        except Exception:
            pass
        try:
            self.release_mouse()
        except Exception:
            pass

        if self._client is not None:
            try:
                self._run_coro_blocking(self._client.disconnect())
            except Exception:
                pass
            self._client = None

        self._stop_loop_thread()

    # ----------------------------------------------------------------------
    # Mouse API (mirrors HIDController.py)
    # ----------------------------------------------------------------------

    def move_mouse(self, x=0, y=0, buttons=0x00):
        """
        Move the mouse and optionally send a button press state.

        Args:
            x (int): Mouse delta X (device clamps to [-127, 127]).
            y (int): Mouse delta Y (device clamps to [-127, 127]).
            buttons (int): Bitmask for buttons (e.g., MOUSE_LEFT, MOUSE_RIGHT).
        """
        if self._client is None:
            return

        with self._lock:
            # If caller provides a new buttons mask, reconcile it
            self._reconcile_mouse_buttons(buttons)

            # Mouse move is high-rate, so do NOT require ACK for every packet.
            self._run_coro_nonblocking(self._client.mouse_move(dx=int(x), dy=int(y), wheel=0, require_ack=False))

    def press_mouse(self, button=MOUSE_LEFT):
        """
        Press a mouse button.

        Args:
            button (int): One of MOUSE_LEFT, MOUSE_RIGHT, MOUSE_MIDDLE.
        """
        if self._client is None:
            return

        with self._lock:
            # Update our pressed mask
            self._pressed_buttons_mask |= int(button)

            # Send explicit press message for the specific button
            btn_name = self._button_mask_to_name(button)
            self._run_coro_blocking(self._client.mouse_press(btn_name, require_ack=True))

    def release_mouse(self):
        """
        Release all mouse buttons.
        """
        if self._client is None:
            return

        with self._lock:
            # Release any buttons we believe are down
            self._release_all_pressed_mouse_buttons()

    # ----------------------------------------------------------------------
    # Keyboard API (mirrors HIDController.py)
    # ----------------------------------------------------------------------

    def press_key(self, key_name, modifier=0x00):
        """
        Press a single key, optionally with modifier.

        Notes:
            Your nRF device currently does NOT implement a modifier field in JSON.
            We handle modifiers by sending the modifier key name as an additional key press
            if you want, but your current gesture system doesn't use modifiers.

        Args:
            key_name (str): The name of the key (e.g., 'w', 'esc', 'a', 'CTRL', 'SHIFT').
            modifier (int): Unused for BLE JSON protocol (kept for interface compatibility).
        """
        if self._client is None:
            return

        # Your device resolves key names; send exactly what it expects.
        # Lowercase letters are accepted if your device key-map supports them.
        keys = [str(key_name)]

        # Require ACK for key press (low-rate)
        self._run_coro_blocking(self._client.keyboard_press(keys, require_ack=True))

    def release_keys(self):
        """
        Send a release signal for all keys.
        """
        if self._client is None:
            return

        self._run_coro_blocking(self._client.keyboard_release_all(require_ack=True))

    def tap_key(self, key_name, modifier=0x00, delay=0.1):
        """
        Tap a key: press and release after a short delay.

        Args:
            key_name (str): Name of the key to tap.
            modifier (int): Unused for BLE JSON protocol (kept for compatibility).
            delay (float): Delay in seconds between press and release.
        """
        self.press_key(key_name, modifier)
        time.sleep(delay)
        self.release_keys()

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _button_mask_to_name(self, mask):
        """
        Convert mouse button bitmask to device button name string.
        """
        if mask == MOUSE_LEFT:
            return "LMB"
        if mask == MOUSE_RIGHT:
            return "RMB"
        if mask == MOUSE_MIDDLE:
            return "MMB"
        # If we get an unknown mask, default to left
        return "LMB"

    def _reconcile_mouse_buttons(self, new_mask):
        """
        Description:
            Reconcile a provided mouse button mask with our current mask.

        Behavior:
            - If new_mask has a button we didn't have, send a press for that button.
            - If we currently have a button that new_mask doesn't, send a release for that button.
        """
        new_mask = int(new_mask)

        # Press newly-down buttons
        newly_pressed = new_mask & (~self._pressed_buttons_mask)
        if newly_pressed:
            for btn_mask in (MOUSE_LEFT, MOUSE_RIGHT, MOUSE_MIDDLE):
                if newly_pressed & btn_mask:
                    btn_name = self._button_mask_to_name(btn_mask)
                    self._run_coro_blocking(self._client.mouse_press(btn_name, require_ack=True))

        # Release newly-up buttons
        newly_released = self._pressed_buttons_mask & (~new_mask)
        if newly_released:
            for btn_mask in (MOUSE_LEFT, MOUSE_RIGHT, MOUSE_MIDDLE):
                if newly_released & btn_mask:
                    btn_name = self._button_mask_to_name(btn_mask)
                    self._run_coro_blocking(self._client.mouse_release(btn_name, require_ack=True))

        self._pressed_buttons_mask = new_mask

    def _release_all_pressed_mouse_buttons(self):
        """
        Release any mouse buttons we believe are currently pressed.
        """
        current = self._pressed_buttons_mask
        if not current:
            return

        for btn_mask in (MOUSE_LEFT, MOUSE_RIGHT, MOUSE_MIDDLE):
            if current & btn_mask:
                btn_name = self._button_mask_to_name(btn_mask)
                self._run_coro_blocking(self._client.mouse_release(btn_name, require_ack=True))

        self._pressed_buttons_mask = 0x00

    def _log(self, msg):
        """
        Simple logger used by BleHidProxyClient.
        """
        print(f"[BLE] {msg}")

    # ----------------------------------------------------------------------
    # Background asyncio loop thread
    # ----------------------------------------------------------------------

    def _start_loop_thread(self):
        """
        Start the background event loop thread exactly once.
        """
        if self._thread is not None:
            return

        def _thread_main():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop_ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_thread_main, daemon=True)
        self._thread.start()
        self._loop_ready.wait(timeout=5.0)

        if self._loop is None:
            raise RuntimeError("Failed to start BLE event loop thread")

    def _stop_loop_thread(self):
        """
        Stop the background loop thread.
        """
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

        self._loop = None
        self._thread = None
        self._loop_ready.clear()

    def _run_coro_blocking(self, coro):
        """
        Schedule a coroutine on the loop and block until it completes.
        """
        if self._loop is None:
            raise RuntimeError("BLE loop not running")

        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def _run_coro_nonblocking(self, coro):
        """
        Schedule a coroutine on the loop and do not block (fire-and-forget).
        """
        if self._loop is None:
            return

        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception:
            pass