################################################################
# ble_communication/protocol.py
#
# Description:
#   Builds the JSON commands your device expects.
#   This file is intentionally dumb: it just creates dicts.
#   No BLE I/O. No retries. No state.
#
################################################################
"""
ble_hid_proxy/protocol.py

Builds the JSON commands your device expects.

This file is intentionally dumb: it just creates dicts.
No BLE I/O. No retries. No state.
"""

from __future__ import annotations


def cmd_auth(token: str) -> dict:
    return {"type": "auth", "token": token}


def cmd_keyboard_write(text: str) -> dict:
    return {"type": "keyboard", "action": "write", "text": text}


def cmd_keyboard_press(keys: list[str]) -> dict:
    return {"type": "keyboard", "action": "press", "keys": keys}


def cmd_keyboard_release(keys: list[str]) -> dict:
    return {"type": "keyboard", "action": "release", "keys": keys}


def cmd_keyboard_release_all() -> dict:
    return {"type": "keyboard", "action": "release_all"}


def cmd_mouse_move(dx: int = 0, dy: int = 0, wheel: int = 0) -> dict:
    return {"type": "mouse", "action": "move", "dx": dx, "dy": dy, "wheel": wheel}


def cmd_mouse_click(button: str = "LMB") -> dict:
    return {"type": "mouse", "action": "click", "button": button}


def cmd_mouse_press(button: str = "LMB") -> dict:
    return {"type": "mouse", "action": "press", "button": button}


def cmd_mouse_release(button: str = "LMB") -> dict:
    return {"type": "mouse", "action": "release", "button": button}