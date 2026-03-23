####################################################################################################
# ble_receiver/keymaps.py
#
# Description: Human-readable input name -> HID constant lookup tables.
#
# Author: Daniel Gebura
####################################################################################################

from adafruit_hid.keycode import Keycode
from adafruit_hid.mouse import Mouse

# --------------------------------------------------------------------------------------------------
# Keyboard Mapping Table
# --------------------------------------------------------------------------------------------------

# This dictionary allows protocol commands to use readable key names such as:
#   "CTRL", "A", "ENTER", "LEFT"
# instead of raw HID numeric values.
KEY_NAME_TO_KEYCODE_MAP = {}


# Build support for letters a..z and A..Z.
# Both lowercase and uppercase string names should resolve to the same keycode.
for ascii_value in range(ord("a"), ord("z") + 1):
    letter = chr(ascii_value)
    KEY_NAME_TO_KEYCODE_MAP[letter] = getattr(Keycode, letter.upper())
    KEY_NAME_TO_KEYCODE_MAP[letter.upper()] = getattr(Keycode, letter.upper())


# Top-row digits.
KEY_NAME_TO_KEYCODE_MAP["0"] = Keycode.ZERO
KEY_NAME_TO_KEYCODE_MAP["1"] = Keycode.ONE
KEY_NAME_TO_KEYCODE_MAP["2"] = Keycode.TWO
KEY_NAME_TO_KEYCODE_MAP["3"] = Keycode.THREE
KEY_NAME_TO_KEYCODE_MAP["4"] = Keycode.FOUR
KEY_NAME_TO_KEYCODE_MAP["5"] = Keycode.FIVE
KEY_NAME_TO_KEYCODE_MAP["6"] = Keycode.SIX
KEY_NAME_TO_KEYCODE_MAP["7"] = Keycode.SEVEN
KEY_NAME_TO_KEYCODE_MAP["8"] = Keycode.EIGHT
KEY_NAME_TO_KEYCODE_MAP["9"] = Keycode.NINE


# Common whitespace / editing keys.
KEY_NAME_TO_KEYCODE_MAP["SPACE"] = Keycode.SPACEBAR
KEY_NAME_TO_KEYCODE_MAP["TAB"] = Keycode.TAB
KEY_NAME_TO_KEYCODE_MAP["ENTER"] = Keycode.ENTER
KEY_NAME_TO_KEYCODE_MAP["RETURN"] = Keycode.ENTER
KEY_NAME_TO_KEYCODE_MAP["ESC"] = Keycode.ESCAPE
KEY_NAME_TO_KEYCODE_MAP["ESCAPE"] = Keycode.ESCAPE
KEY_NAME_TO_KEYCODE_MAP["BACKSPACE"] = Keycode.BACKSPACE
KEY_NAME_TO_KEYCODE_MAP["DELETE"] = Keycode.DELETE


# Arrow keys.
KEY_NAME_TO_KEYCODE_MAP["UP"] = Keycode.UP_ARROW
KEY_NAME_TO_KEYCODE_MAP["DOWN"] = Keycode.DOWN_ARROW
KEY_NAME_TO_KEYCODE_MAP["LEFT"] = Keycode.LEFT_ARROW
KEY_NAME_TO_KEYCODE_MAP["RIGHT"] = Keycode.RIGHT_ARROW


# Navigation keys.
KEY_NAME_TO_KEYCODE_MAP["HOME"] = Keycode.HOME
KEY_NAME_TO_KEYCODE_MAP["END"] = Keycode.END
KEY_NAME_TO_KEYCODE_MAP["PAGE_UP"] = Keycode.PAGE_UP
KEY_NAME_TO_KEYCODE_MAP["PAGE_DOWN"] = Keycode.PAGE_DOWN


# Modifier keys.
KEY_NAME_TO_KEYCODE_MAP["CTRL"] = Keycode.CONTROL
KEY_NAME_TO_KEYCODE_MAP["CONTROL"] = Keycode.CONTROL
KEY_NAME_TO_KEYCODE_MAP["SHIFT"] = Keycode.SHIFT
KEY_NAME_TO_KEYCODE_MAP["ALT"] = Keycode.ALT
KEY_NAME_TO_KEYCODE_MAP["GUI"] = Keycode.GUI


# --------------------------------------------------------------------------------------------------
# Mouse Button Mapping Table
# --------------------------------------------------------------------------------------------------

MOUSE_BUTTON_TO_BUTTON_VALUE_MAP = {
    "LMB": Mouse.LEFT_BUTTON,
    "RMB": Mouse.RIGHT_BUTTON,
    "MMB": Mouse.MIDDLE_BUTTON,
}
