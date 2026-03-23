####################################################################################################
# ble_receiver/hid_controller.py
#
# Description: USB HID output layer. This module owns all keyboard and mouse interaction with the host 
# computer. It provides a single, easy-to-reason-about class that converts validated protocol commands 
# into concrete HID actions.
#
# Author: Daniel Gebura
####################################################################################################
import usb_hid

from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keyboard_layout_us import KeyboardLayoutUS
from adafruit_hid.mouse import Mouse

from ble_receiver.config import MOUSE_DELTA_MAX
from ble_receiver.config import MOUSE_DELTA_MIN
from ble_receiver.keymaps import KEY_NAME_TO_KEYCODE_MAP
from ble_receiver.keymaps import MOUSE_BUTTON_TO_BUTTON_VALUE_MAP

class HIDController:
    """
    Description:
        High-level wrapper around CircuitPython USB HID keyboard and mouse helpers.
    """

    def __init__(self):
        """
        Description:
            Create and bind the USB keyboard / mouse HID device helpers.

        Notes:
            - `usb_hid.devices` exposes the HID interfaces currently available over USB.
            - `KeyboardLayoutUS` converts text strings into the correct key sequences for a US layout.
        """
        self.keyboard = Keyboard(usb_hid.devices)
        self.keyboard_layout = KeyboardLayoutUS(self.keyboard)
        self.mouse = Mouse(usb_hid.devices)

    @staticmethod
    def _clamp(value, min_value, max_value):
        """
        Description:
            Clamp an integer to an inclusive minimum / maximum range.
        """
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value

    @staticmethod
    def _resolve_keycodes_from_names(key_name_list):
        """
        Description:
            Convert a list of protocol key names into actual HID keycode constants.

        Args:
            key_name_list:
                Iterable of key names such as ["CTRL", "ALT", "DELETE"].

        Returns:
            List of resolved HID keycode integers.

        Raises:
            ValueError:
                Raised when a key name is missing, invalid, or unknown.
        """
        if not isinstance(key_name_list, list) or not key_name_list:
            raise ValueError("keyboard action requires non-empty 'keys' list")

        resolved_keycodes = []

        for key_name in key_name_list:
            normalized_key_name = str(key_name).upper()
            keycode = KEY_NAME_TO_KEYCODE_MAP.get(normalized_key_name)

            if keycode is None:
                raise ValueError("Unknown key name: {}".format(key_name))

            resolved_keycodes.append(keycode)

        return resolved_keycodes

    def handle_keyboard_command(self, cmd):
        """
        Description:
            Execute a validated keyboard command.

        Supported actions:
            - write
            - press
            - release
            - release_all

        Expected fields:
            For action == "write":
                {"type":"keyboard", "action":"write", "text":"hello"}

            For action == "press" or "release":
                {"type":"keyboard", "action":"press", "keys":["CTRL", "C"]}
        """
        action = cmd.get("action")

        if action == "write":
            text = cmd.get("text", "")
            if not isinstance(text, str):
                raise ValueError("keyboard.write requires 'text' string")

            self.keyboard_layout.write(text)
            return

        if action == "press":
            keycode_list = self._resolve_keycodes_from_names(cmd.get("keys", []))
            self.keyboard.press(*keycode_list)
            return

        if action == "release":
            keycode_list = self._resolve_keycodes_from_names(cmd.get("keys", []))
            self.keyboard.release(*keycode_list)
            return

        if action == "release_all":
            self.keyboard.release_all()
            return

        raise ValueError("Unknown keyboard action: {!r}".format(action))

    def handle_mouse_command(self, cmd):
        """
        Description:
            Execute a validated mouse command.

        Supported actions:
            - move
            - click
            - press
            - release
        """
        action = cmd.get("action")

        if action == "move":
            dx = int(cmd.get("dx", 0))
            dy = int(cmd.get("dy", 0))
            wheel = int(cmd.get("wheel", 0))

            # HID mouse reports use signed 8-bit deltas, so all values must be clamped.
            dx = self._clamp(dx, MOUSE_DELTA_MIN, MOUSE_DELTA_MAX)
            dy = self._clamp(dy, MOUSE_DELTA_MIN, MOUSE_DELTA_MAX)
            wheel = self._clamp(wheel, MOUSE_DELTA_MIN, MOUSE_DELTA_MAX)

            self.mouse.move(dx, dy, wheel)
            return

        if action in ("click", "press", "release"):
            button_name = str(cmd.get("button", "LMB")).upper()
            button_value = MOUSE_BUTTON_TO_BUTTON_VALUE_MAP.get(button_name)

            if button_value is None:
                raise ValueError("mouse button must be one of: LMB|RMB|MMB")

            if action == "click":
                self.mouse.click(button_value)
            elif action == "press":
                self.mouse.press(button_value)
            else:
                self.mouse.release(button_value)
            return

        raise ValueError("Unknown mouse action: {!r}".format(action))

    def route_command(self, cmd):
        """
        Description:
            Route a parsed JSON command to the correct HID handler.
        """
        command_type = cmd.get("type")

        if command_type == "keyboard":
            self.handle_keyboard_command(cmd)
            return

        if command_type == "mouse":
            self.handle_mouse_command(cmd)
            return

        raise ValueError("Command 'type' must be 'keyboard' or 'mouse'")

    def cleanup_all_inputs(self):
        """
        Description:
            Force-release all keyboard keys and mouse buttons.
        """
        try:
            self.keyboard.release_all()
        except Exception:
            pass

        try:
            self.mouse.release(Mouse.LEFT_BUTTON)
            self.mouse.release(Mouse.RIGHT_BUTTON)
            self.mouse.release(Mouse.MIDDLE_BUTTON)
        except Exception:
            pass
