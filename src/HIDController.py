################################################################
# HIDController.py
#
# Description: HID Controller for Jetson Nano USB Gadget. This script abstracts 
# low-level HID input report writing into simple Python function calls for 
# mouse movement, clicks, and keyboard input (e.g., WASD + ESC).
#
# Author: Daniel Gebura
################################################################

import time
import os

# Device file paths for HID output
MOUSE_DEV = "/dev/hidg0"      # Mouse HID report interface
KEYBOARD_DEV = "/dev/hidg1"   # Keyboard HID report interface

# HID Keycodes (from HID Usage Tables)
KEYCODES = {
    "a": 0x04,
    "d": 0x07,
    "s": 0x16,
    "w": 0x1A,
    "esc": 0x29
}

# Mouse Button Masks
MOUSE_LEFT   = 0x01
MOUSE_RIGHT  = 0x02
MOUSE_MIDDLE = 0x04

class HIDController:
    """
    Abstraction class to control HID mouse and keyboard events via USB gadget.
    """

    def __init__(self, mouse_dev=MOUSE_DEV, keyboard_dev=KEYBOARD_DEV):
        """
        Initialize HIDController with file paths for mouse and keyboard HID devices.

        Args:
            mouse_dev (str): Path to the HID gadget file for mouse (default: '/dev/hidg0').
            keyboard_dev (str): Path to the HID gadget file for keyboard (default: '/dev/hidg1').
        """
        self.mouse_dev = mouse_dev
        self.keyboard_dev = keyboard_dev

    # MOUSE FUNCTIONS -----------------------------------------------

    def move_mouse(self, x=0, y=0, buttons=0x00):
        """
        Move the mouse and optionally send a button press.

        Args:
            x (int): Number of pixels to move in the x-direction (-127 to 127).
            y (int): Number of pixels to move in the y-direction (-127 to 127).
            buttons (int): Bitmask for buttons (e.g., MOUSE_LEFT, MOUSE_RIGHT).
        """
        report = bytes([buttons & 0x07, x & 0xFF, y & 0xFF])  # 3-byte mouse report
        self._write_report(self.mouse_dev, report)

        def press_mouse(self, button=MOUSE_LEFT):
        """
        Press a mouse button.

        Args:
            button (int): One of the MOUSE_* constants.
        """
            self.move_mouse(buttons=button)

        def release_mouse(self):
            """
            Release all mouse buttons.
            """
            self.move_mouse(buttons=0x00)

    # KEYBOARD FUNCTIONS --------------------------------------------

    def press_key(self, key_name, modifier=0x00):
        """
        Press a single key, optionally with a modifier (e.g., Shift).

        Args:
            key_name (str): The name of the key (e.g., 'w', 'esc').
            modifier (int): Modifier byte (e.g., 0x02 for Shift).
        
        Raises:
            ValueError: If the key name is not mapped.
        """
        keycode = KEYCODES.get(key_name.lower())
        if keycode is None:
            raise ValueError(f"Unsupported key: {key_name}")

        # HID keyboard report: [modifier, reserved, key1, key2, key3, key4, key5, key6]
        report = bytes([modifier, 0x00, keycode, 0, 0, 0, 0, 0])
        self._write_report(self.keyboard_dev, report)

    def release_keys(self):
        """
        Send a release signal for all keys.
        """
        self._write_report(self.keyboard_dev, bytes([0]*8))

    def tap_key(self, key_name, modifier=0x00, delay=0.1):
        """
        Tap a key: press and release after a short delay.

        Args:
            key_name (str): Name of the key to tap.
            modifier (int): Modifier byte (optional).
            delay (float): Delay in seconds between press and release.
        """
        self.press_key(key_name, modifier)
        time.sleep(delay)
        self.release_keys()

    # INTERNAL HELPER -----------------------------------------------

    def _write_report(self, device, report):
        """
        Write a binary report to the specified HID device.

        Args:
            device (str): Path to the HID gadget file.
            report (bytes): Report bytes to send.

        Logs:
            Prints an error if the device cannot be accessed or written.
        """
        try:
            with open(device, "wb") as f:
                f.write(report)
        except FileNotFoundError:
            print(f"[ERROR] HID device not found: {device}")
        except PermissionError:
            print(f"[ERROR] Permission denied writing to {device}")
        except Exception as e:
            print(f"[ERROR] Failed to write to {device}: {e}")