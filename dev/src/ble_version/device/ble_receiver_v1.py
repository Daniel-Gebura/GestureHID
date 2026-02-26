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
MOUSE_DELTA_MIN = -127              # Minimum value for mouse movement deltas and wheel
MOUSE_DELTA_MAX = 127               # Maximum value for mouse movement deltas and wheel
IDLE_SLEEP_S = 0.01                 # Delay (seconds) between BLE polls

REQUIRE_BONDING = True              # Require BLE pairing/bonding before accepting commands
AUTH_TOKEN      = "CHANGE_ME"       # Lightweight shared secret for command authorization


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
# Connection Security State
# ----------------------------------------------------------------------

ALLOWED_CENTRAL_ADDRESS = None   # Cache the first connected central address until reset
SESSION_AUTHENTICATED  = False  # Require lightweight auth before processing HID commands
WAS_CONNECTED = False          # Track if we've ever been connected since last reset (for allow-list logic)


# ----------------------------------------------------------------------
# Helper Function Definitions
# ----------------------------------------------------------------------

def _get_active_connection():
    """
    Description:
        Return the first active BLE connection, or None if not connected.

    Returns:
        A BLEConnection or None
    """
    try:
        conns = ble.connections
        if conns and conns[0] is not None and conns[0].connected:
            return conns[0]
    except Exception:
        pass
    return None


def _format_address_hex(addr_obj):
    """
    Description:
        Convert a _bleio.Address (or similar) into a stable hex string.

    Notes:
        We avoid relying on any one property name too strongly; CircuitPython BLE APIs
        have evolved and some attributes differ across versions.

    Returns:
        A string like "AA:BB:CC:DD:EE:FF" or None if unavailable.
    """
    # 1) Best case: str(address) returns human-readable address on many builds
    try:
        s = str(addr_obj)
        if s and ":" in s:
            return s.upper()
    except Exception:
        pass

    # 2) Try raw bytes on address-like objects (rarely needed)
    try:
        b = bytes(addr_obj)
        if b and len(b) == 6:
            return ":".join("{:02X}".format(x) for x in b[::-1])
    except Exception:
        pass

    return None


def _get_connection_address_str(connection):
    """
    Description:
        Get the peer (central) address string for a given BLEConnection.

    Returns:
        Address string "AA:BB:CC:DD:EE:FF" or None if not available.
    """
    # The BLEConnection wraps an internal _bleio.Connection. Many builds expose it as _bleio_connection.
    try:
        bleio_conn = getattr(connection, "_bleio_connection", None)
        if bleio_conn is not None:
            addr = getattr(bleio_conn, "address", None)
            if addr is not None:
                return _format_address_hex(addr)
    except Exception:
        pass

    return None


def _enforce_connection_security(connection):
    """
    Description:
        Enforce bonding + allow-list (cached central address) for the active connection.

    Behavior:
        - If bonding is required, attempt to pair(bond=True). Disconnect if pairing fails.
        - If this is the first-ever central since reset, cache its address and accept it.
        - If a different central connects later, immediately disconnect it.

    Returns:
        True if the connection is allowed, False if it was rejected.
    """
    global ALLOWED_CENTRAL_ADDRESS

    # 1) Require pairing/bonding (best effort, but enforced if enabled)
    if REQUIRE_BONDING:
        try:
            if not connection.paired:
                connection.pair(bond=True)
        except Exception:
            try:
                connection.disconnect()
            except Exception:
                pass
            return False

        # If pairing didn't "stick", treat as failure
        try:
            if not connection.paired:
                connection.disconnect()
            return bool(connection.paired)
        except Exception:
            try:
                connection.disconnect()
            except Exception:
                pass
            return False

    # 2) Allow-list enforcement (cache first central address until reset)
    peer_addr = _get_connection_address_str(connection)
    if peer_addr is None:
        # If we can't read the address, we cannot safely enforce allow-listing.
        # In this case, fail closed (disconnect) because you explicitly asked to refuse others.
        try:
            connection.disconnect()
        except Exception:
            pass
        return False

    if ALLOWED_CENTRAL_ADDRESS is None:
        ALLOWED_CENTRAL_ADDRESS = peer_addr
        return True

    if peer_addr != ALLOWED_CENTRAL_ADDRESS:
        try:
            connection.disconnect()
        except Exception:
            pass
        return False

    return True


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


def _clamp(value, min_value, max_value):
    """
    Description:
        Clamp an integer value into the inclusive range [min_value, max_value].

    Returns:
        Clamped integer.
    """
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


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

    if action == "move":
        dx = int(cmd.get("dx", 0))
        dy = int(cmd.get("dy", 0))
        wheel = int(cmd.get("wheel", 0))
        # Clamp to valid USB HID boot protocol range
        dx = _clamp(dx, MOUSE_DELTA_MIN, MOUSE_DELTA_MAX)
        dy = _clamp(dy, MOUSE_DELTA_MIN, MOUSE_DELTA_MAX)
        wheel = _clamp(wheel, MOUSE_DELTA_MIN, MOUSE_DELTA_MAX)
        mouse.move(dx=dx, dy=dy, wheel=wheel)
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


def _cleanup_hid_state():
    """
    Description:
        Force-release all keyboard keys and mouse buttons.
        This prevents stuck keys/buttons if BLE disconnects mid-command.
    """
    try:
        keyboard.release_all()
    except Exception:
        pass

    try:
        # Release all mouse buttons explicitly
        mouse.release(Mouse.LEFT_BUTTON)
        mouse.release(Mouse.RIGHT_BUTTON)
        mouse.release(Mouse.MIDDLE_BUTTON)
    except Exception:
        pass


def _handle_auth_command(cmd):
    """
    Description:
        Lightweight session authorization.

    Supported patterns:
        1) Explicit auth command:
            {"type":"auth","token":"..."}
        2) Inline auth field on any command BEFORE authenticated:
            {"type":"keyboard", ... , "auth":"..."}

    Returns:
        True if authentication succeeded, False otherwise.
    """
    global SESSION_AUTHENTICATED

    # Already authorized this session
    if SESSION_AUTHENTICATED:
        return True

    # Explicit auth command
    if cmd.get("type") == "auth":
        token = cmd.get("token", None)
        if isinstance(token, str) and token == AUTH_TOKEN:
            SESSION_AUTHENTICATED = True
            return True
        return False

    # Inline auth field on first command(s)
    token = cmd.get("auth", None)
    if isinstance(token, str) and token == AUTH_TOKEN:
        SESSION_AUTHENTICATED = True
        return True

    return False


# ----------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------

# Outer loop runs forever
while True:

    # Detect disconnect transition and cleanup
    if WAS_CONNECTED and not ble.connected:
        _cleanup_hid_state()
        SESSION_AUTHENTICATED = False

    WAS_CONNECTED = ble.connected
    # If disconnected, (re)start advertising and wait for a central
    if not ble.connected:
        # Reset per-connection auth state (but keep allow-list cached until reset)
        SESSION_AUTHENTICATED = False

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

        # 4. Enforce bonding + allow-listing for the newly connected central
        conn = _get_active_connection()
        if conn is None:
            # If we cannot resolve the connection object, fail closed.
            try:
                ble.stop_advertising()
            except Exception:
                pass
            continue

        if not _enforce_connection_security(conn):
            # Rejected: drop back to advertising loop
            continue

        print("Central connected (secured).")

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

            # Enforce lightweight auth gating before any HID actions
            if not _handle_auth_command(cmd):
                try:
                    uart.write(b'{"ok":false,"error":"unauthorized"}\n')
                except Exception:
                    pass
                continue

            # If this was an explicit auth command, acknowledge and continue
            if cmd.get("type") == "auth":
                try:
                    uart.write(b'{"ok":true,"authed":true}\n')
                except Exception:
                    pass
                continue

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