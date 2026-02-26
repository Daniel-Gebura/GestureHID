################################################################
# ble_receiver_v1.py  (Referred to as "code.py" on Adafruit nRF52840 USB Stick)
#
# Description: BLE (Nordic UART) JSON control -> USB HID Keyboard/Mouse.
# Receives newline-delimited JSON commands over BLE UART and executes
# keyboard/mouse actions via USB HID on the plugged-in computer.
#
# Author: Daniel Gebura
################################################################

import json     # adafruit JSON module for parsing JSON command strings
import time     # adafruit time module for sleeps and timing control
import usb_hid  # adafruit USB HID module to access USB HID devices (keyboard/mouse)

# Import BLE helpers from included adafruit_ble library
from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

# Import HID helpers from included adafruit_ble library
from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keyboard_layout_us import KeyboardLayoutUS
from adafruit_hid.keycode import Keycode
from adafruit_hid.mouse import Mouse


# ----------------------------------------------------------------------
# Configuration Variables
# ----------------------------------------------------------------------

DEVICE_NAME  = "HID-Proxy-Control"  # BLE advertising name
JSON_MAX_LEN = 512                  # Safety cap on incoming line size
IDLE_SLEEP_S = 0.001                # Delay (seconds) between BLE polls


# ----------------------------------------------------------------------
# HID Setup
# ----------------------------------------------------------------------

# 1. Initialize HID devices (Keyboard and mouse are separate)
keyboard = Keyboard(usb_hid.devices)          # Create USB keyboard HID object
keyboard_layout = KeyboardLayoutUS(keyboard)  # Attach US layout to keyboard
mouse = Mouse(usb_hid.devices)                # Create USB mouse HID object

# 2. Create key mapping dictionary for resolving common key names to actual Keycode values
KEY_NAME_TO_KEYCODE_MAP = {
    # Letters a..z mapped to their uppercase Keycode equivalents
    **{
        chr(ascii_code): getattr(Keycode, chr(ascii_code).upper())
        for ascii_code in range(ord("a"), ord("z") + 1)
    },
    # digits 0..9 (top row)
    **{
        str(digit): getattr(Keycode, f"NUMBER_{digit}")
        for digit in range(10)
    },
    # whitespace & common keys
    "SPACE": Keycode.SPACEBAR,
    "TAB": Keycode.TAB,
    "ENTER": Keycode.ENTER,
    "RETURN": Keycode.ENTER,
    "ESC": Keycode.ESCAPE,
    "ESCAPE": Keycode.ESCAPE,
    "BACKSPACE": Keycode.BACKSPACE,
    "DELETE": Keycode.DELETE,
    # arrows
    "UP": Keycode.UP_ARROW,
    "DOWN": Keycode.DOWN_ARROW,
    "LEFT": Keycode.LEFT_ARROW,
    "RIGHT": Keycode.RIGHT_ARROW,
    # navigation
    "HOME": Keycode.HOME,
    "END": Keycode.END,
    "PAGE_UP": Keycode.PAGE_UP,
    "PAGE_DOWN": Keycode.PAGE_DOWN,
    # modifiers
    "CTRL": Keycode.CONTROL,
    "CONTROL": Keycode.CONTROL,
    "SHIFT": Keycode.SHIFT,
    "ALT": Keycode.ALT,
    "GUI": Keycode.GUI,  # Windows / Command key
    # function keys F1..F24
    **{
        f"F{f_index}": getattr(Keycode, f"F{f_index}")
        for f_index in range(1, 25)
    },
}

# 3. Create mouse button mapping dictionary
MOUSE_BUTTON_TO_BUTTON_VALUE_MAP = {
    "LMB": Mouse.LEFT_BUTTON,
    "RMB": Mouse.RIGHT_BUTTON,
    "MMB": Mouse.MIDDLE_BUTTON,
}


# ----------------------------------------------------------------------
# BLE Setup
# ----------------------------------------------------------------------

# Initialize the BLE radio UART service and advertisement
ble = BLERadio()                                    # Create BLE radio object
ble.name = DEVICE_NAME                              # Set the BLE advertising name
uart = UARTService()                                # Create the Nordic UART BLE service object
uart_advertisement = ProvideServicesAdvertisement(uart)  # Create advertisement that exposes the UART service


# ----------------------------------------------------------------------
# Helper Function Definitions
# ----------------------------------------------------------------------

def _resolve_keycodes_from_names(key_name_list):
    """
    Description:
        Resolve a list of key name strings into a list of Keycode integer values.

    Args:
        key_name_list: A list of key name strings (for example: ["CTRL", "ALT", "DEL"]).

    Returns:
        A list of integer Keycode values corresponding to the provided names.

    Raises:
        ValueError: If any element is not a string or the key name is unknown.
    """
    resolved_keycodes = []  # Initialize list to hold resolved keycodes

    # Iterate over the incoming key names and resolve each to a keycode value
    for key_name in key_name_list:
        # Validate that this key name is a string
        if not isinstance(key_name, str):
            raise ValueError("All key names must be strings")

        # Lookup the keycode for this normalized key name
        keycode_value = KEY_NAME_TO_KEYCODE_MAP.get(key_name.upper())

        # Validate that the key name was found in dictionary
        if keycode_value is None:
            raise ValueError(f"Unknown key: {key_name!r}")

        # Append the resolved keycode value to the output list
        resolved_keycodes.append(keycode_value)

    return resolved_keycodes


def _handle_keyboard_command(cmd: dict) -> None:
    """
    Description:
        Execute a keyboard-related JSON command.

    Args:
        cmd: Parsed JSON dict with fields:
             - "action": "write" | "press" | "release" | "release_all"
             - "text": string to type when action is "write"
             - "keys": list of key name strings for "press" or "release" actions

    Returns:
        None

    Raises:
        ValueError: On invalid action or bad/missing fields.
    """
    # Extract the action field for this command
    action = cmd.get("action")

    # Write: Type literal text via layout mapping
    if action == "write":
        text = cmd.get("text", "")  # Get text field from cmd (default to empty string if missing)
        if not isinstance(text, str):  # Validate that text is a string
            raise ValueError("keyboard.write requires 'text' string")
        keyboard_layout.write(text)  # Use layout to write the full text to the pc
        return

    # Press: Press one or more keys
    if action == "press":
        keycode_list = _resolve_keycodes_from_names(cmd.get("keys", []))  # Resolve key names to keycodes
        keyboard.press(*keycode_list)  # Press all specified keys
        return 

    # Release: Release one or more keys
    if action == "release":
        keycode_list = _resolve_keycodes_from_names(cmd.get("keys", []))  # Resolve key names to keycodes
        keyboard.release(*keycode_list)  # Release all specified keys
        return

    # Release All: Release all currently pressed keys
    if action == "release_all":
        keyboard.release_all()  # Release all currently pressed keys
        return

    # If we reach here, the action was unknown
    raise ValueError(f"Unknown keyboard action: {action!r}")


def _handle_mouse_command(cmd: dict) -> None:
    """
    Description:
        Execute a mouse-related JSON command.

    Args:
        cmd: Parsed JSON dict with fields:
            - "action": "move" | "click" | "press" | "release"
            - "dx": horizontal movement for "move"
            - "dy": vertical movement for "move"
            - "wheel": wheel delta for "move"
            - "button": "LMB" | "RMB" | "MMB" for button actions

    Returns:
        None

    Raises:
        ValueError: On invalid action or button value.
    """
    # Extract the action field for this command
    action = cmd.get("action")

    # Move: Move the cursor and/or wheel
    if action == "move":
        dx = int(cmd.get("dx", 0))  # Get horizontal delta (default 0)
        dy = int(cmd.get("dy", 0))  # Get vertical delta (default 0)
        wheel = int(cmd.get("wheel", 0))  # Get wheel delta (default 0)
        mouse.move(dx=dx, dy=dy, wheel=wheel)  # Move the mouse accordingly
        return

    # Button Commands: Click, press, or release a button
    if action in ("click", "press", "release"):
        btn_name = str(cmd.get("button", "LMB")).upper()  # Get the button name (default left mouse button)
        btn = MOUSE_BUTTON_TO_BUTTON_VALUE_MAP.get(btn_name)  # Resolve button name to actual value
        if btn is None:  # Validate that the button name is known
            raise ValueError("mouse button must be one of: LMB|RMB|MMB")
        if action == "click":
            mouse.click(btn)  # Click the specified button
        elif action == "press":
            mouse.press(btn)  # Press the specified button
        else:
            mouse.release(btn)  # Release the specified button
        return

    # If we reach here, the action was unknown
    raise ValueError(f"Unknown mouse action: {action!r}")


def _route_command(cmd):
    """
    Description:
        Route a parsed JSON command to the appropriate keyboard or mouse handler.

    Args:
        cmd: Parsed JSON dict with at least a "type" field.

    Returns:
        None

    Raises:
        ValueError: If the command type is unsupported.
    """
    command_type = cmd.get("type")  # Extract the command type
    if command_type == "keyboard":
        _handle_keyboard_command(cmd)  # Pass this command to the keyboard handler
    elif command_type == "mouse":
        _handle_mouse_command(cmd)  # Pass this command to the mouse handler
    else:
        raise ValueError("Command 'type' must be 'keyboard' or 'mouse'")


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------

# Outer loop runs forever
while True:
    # If disconnected, (re)start advertising and wait for a central
    if not ble.connected:
        # 1. Start advertising (ignore errors)
        try:
            ble.start_advertising(uart_advertisement)
        except Exception:
            # Ignore "already advertising" or transient stack issues
            pass

        # 2. Wait here until a central connects
        while not ble.connected:
            time.sleep(IDLE_SLEEP_S)

        # 3. Stop advertising once connected (ignore errors)
        try:
            ble.stop_advertising()
        except Exception:
            # Ignore failures in stopping advertising
            pass
        print("Central connected.")

    # While connected, continually read and process newline-delimited JSON commands
    while ble.connected:
        # 1. Read a line of input from the UART service (May be None)
        raw_line_bytes = uart.readline()
        if not raw_line_bytes:        # No data received this iteration
            time.sleep(IDLE_SLEEP_S)  # Sleep briefly and skip to next iteration
            continue

        # 2. Drop oversized payloads for safety
        if len(raw_line_bytes) > JSON_MAX_LEN:
            continue

        # 3. Try to decode, parse, and route the command
        try:
            # Decode bytes -> str, strip whitespace/newline, parse JSON
            cmd = json.loads(raw_line_bytes.decode("utf-8").strip())

            # Execute and acknowledge success
            _route_command(cmd)
            try:
                uart.write(b'{"ok":true}\n')  # Send a success acknowledgment back to the host
            except Exception:
                # Best-effort write; OK if notify fails
                pass

        # Catch any exceptions during command processing
        except Exception as e:
            # Structured error back to host (best-effort)
            try:
                uart.write(
                    ('{"ok":false,"error":' + json.dumps(str(e)) + "}\n").encode("utf-8")
                )
            except Exception:
                # If TX back to host fails, just continue
                pass