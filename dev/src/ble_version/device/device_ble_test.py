################################################################
# code.py
# Minimal BLE-connected → USB HID demo.
# - Types "a" when not connected, "b" when connected over BLE.
# - Uses Nordic UART service only for connection state.
# - Uses KeyboardLayoutUS to reliably emit characters.
#
# Author: Daniel
# Date: 2025-09-30
################################################################

import time
import usb_hid

from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keyboard_layout_us import KeyboardLayoutUS
from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DEVICE_NAME: str = "HID-Proxy-Basic"  # BLE advertising name
TYPE_INTERVAL_S: float = 2.5          # Keypress interval (seconds)


# ----------------------------------------------------------------------
# Hardware/stack initialization
# ----------------------------------------------------------------------

def init_keyboard_and_layout() -> tuple[Keyboard, KeyboardLayoutUS]:
    """
    Description:
        Initialize the USB HID keyboard and bind a US layout mapper.

    Args:
        None

    Returns:
        (Keyboard, KeyboardLayoutUS):
            The raw keyboard interface and its layout helper.
    """
    kbd = Keyboard(usb_hid.devices)
    layout = KeyboardLayoutUS(kbd)
    return kbd, layout


def init_ble() -> tuple[BLERadio, UARTService, ProvideServicesAdvertisement]:
    """
    Description:
        Initialize the BLE radio, set its advertising name, create a
        Nordic UART service (used only for connection state), and an
        advertisement object.

    Args:
        None

    Returns:
        (BLERadio, UARTService, ProvideServicesAdvertisement):
            The BLE radio, UART service, and advertisement.
    """
    ble = BLERadio()
    ble.name = DEVICE_NAME
    uart = UARTService()
    advertisement = ProvideServicesAdvertisement(uart)
    return ble, uart, advertisement


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def type_char(layout: KeyboardLayoutUS, ch: str) -> None:
    """
    Description:
        Type a single character using the US keyboard layout, with
        basic error handling.

    Args:
        layout: KeyboardLayoutUS object bound to the keyboard.
        ch:     Single character string to type.

    Returns:
        None
    """
    try:
        layout.write(ch)
    except Exception as e:
        # Non-fatal: happens if no text field is focused, etc.
        print(f"Keyboard write error ({ch}):", e)


# ----------------------------------------------------------------------
# Main run loop
# ----------------------------------------------------------------------

def main() -> None:
    """
    Description:
        Main loop controlling BLE advertising and typing behavior.

        - While not connected over BLE:
            Advertise and type 'a' at the configured interval.
        - Once connected over BLE:
            Stop advertising and type 'b' at the configured interval.

    Args:
        None

    Returns:
        None
    """
    _kbd, layout = init_keyboard_and_layout()
    ble, _uart, advertisement = init_ble()

    advertising = False
    print("Starting. Will type 'a' when not connected; 'b' when connected.")

    while True:
        if not ble.connected:
            # If not yet advertising, start advertising.
            if not advertising:
                try:
                    ble.start_advertising(advertisement)
                    advertising = True
                    print("Advertising as:", DEVICE_NAME)
                except Exception as e:
                    # If already advertising or a transient error occurs.
                    print("Advertising error:", e)

            # While not connected, type 'a'.
            type_char(layout, "a")
            time.sleep(TYPE_INTERVAL_S)

        else:
            # If connected, stop advertising once and switch to typing 'b'.
            if advertising:
                try:
                    ble.stop_advertising()
                except Exception:
                    pass
                advertising = False
                print("Central connected. Typing 'b'.")

            type_char(layout, "b")
            time.sleep(TYPE_INTERVAL_S)


# ----------------------------------------------------------------------
# Execute immediately on import (CircuitPython convention)
# ----------------------------------------------------------------------

main()