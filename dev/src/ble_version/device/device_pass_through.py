################################################################
# code.py
# BLE UART -> USB HID passthrough (single-char).
# Receives bytes over Nordic UART (BLE) and types them via USB HID
# into the computer this stick is plugged into. Printable ASCII
# (32..126) is emitted; LF/CR/newlines and non-printables are ignored.
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


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DEVICE_NAME: str = "HID-Proxy-Simple"  # BLE advertising name
LOOP_SLEEP_S: float = 0.002            # Small delay to yield BLE/USB stacks


# ----------------------------------------------------
# Initialization
# ----------------------------------------------------

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
        Initialize BLE radio with the configured device name, create a Nordic
        UART Service, and prepare an advertisement for that service.

    Args:
        None

    Returns:
        (BLERadio, UARTService, ProvideServicesAdvertisement):
            The BLE radio, UART service, and advertisement object.
    """
    ble = BLERadio()
    ble.name = DEVICE_NAME
    uart = UARTService()
    advertisement = ProvideServicesAdvertisement(uart)
    return ble, uart, advertisement


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def type_printable_bytes(layout: KeyboardLayoutUS, data: bytes) -> None:
    """
    Description:
        Iterate over received bytes and emit only printable ASCII characters
        (range 32..126). Newlines (LF 10, CR 13) and other non-printables
        are ignored.

    Args:
        layout:
            KeyboardLayoutUS bound to the HID keyboard.
        data:
            Bytes read from BLE UART.

    Returns:
        None
    """
    # Process each byte individually to keep behavior simple and robust.
    for b in data:
        # Ignore LF/CR delimiters used by many UART senders.
        if b in (10, 13):
            continue
        # Emit only printable ASCII.
        if 32 <= b <= 126:
            ch = chr(b)
            try:
                layout.write(ch)
            except Exception as e:
                # Non-fatal (e.g., host has no focused input field).
                print(f"Type error ({ch!r}):", e)
        # All other bytes are dropped in this minimal version.


# ----------------------------------------------------
# Main run loop
# ----------------------------------------------------

def main() -> None:
    """
    Description:
        Advertise a BLE Nordic UART service and act as a byte→char passthrough:
        - While disconnected: ensure advertising stays active.
        - When connected: stop advertising and drain UART bytes, emitting
          printable ASCII to USB HID.

    Args:
        None

    Returns:
        None
    """
    _kbd, layout = init_keyboard_and_layout()
    ble, uart, advertisement = init_ble()

    advertising = False
    print("Starting BLE HID passthrough… Advertising as:", DEVICE_NAME)

    # Continuous operation (CircuitPython main script pattern).
    while True:
        # Start advertising if disconnected and not already advertising.
        if not ble.connected and not advertising:
            try:
                ble.start_advertising(advertisement)
                advertising = True
                print("Advertising…")
            except Exception as e:
                # If already advertising or transient stack error, continue.
                print("Advertising error:", e)

        # Handle connected state.
        if ble.connected:
            # Stop advertising once on connection.
            if advertising:
                try:
                    ble.stop_advertising()
                except Exception:
                    # Harmless if advertising already stopped.
                    pass
                advertising = False
                print("Central connected. Ready to type received chars.")

            # Read as many bytes as are available and type them.
            try:
                # Prefer in_waiting if available (newer bundles).
                n = uart.in_waiting  # type: ignore[attr-defined]
            except AttributeError:
                # Fallback for older bundles that lack in_waiting.
                n = 1

            if n:
                try:
                    data = uart.read(n)  # Returns bytes or None
                except Exception as e:
                    print("UART read error:", e)
                    data = None

                if data:
                    type_printable_bytes(layout, data)

        # Yield to BLE/USB stacks and avoid busy-waiting.
        time.sleep(LOOP_SLEEP_S)


# ----------------------------------------------------
# Execute immediately on import (CircuitPython convention)
# ----------------------------------------------------

main()
