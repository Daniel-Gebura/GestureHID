################################################################
# host_ble_test.py
# Minimal host script to find and connect to the nRF52840 HID Proxy.
# - Scans for a device named "HID-Proxy-Basic"
# - Connects and stays connected until terminated
# - Prints connection status updates
# Author: Daniel
# Date: 2025-09-30
################################################################

import asyncio
from bleak import BleakScanner, BleakClient

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

DEVICE_NAME: str = "HID-Proxy-Basic"

# Nordic UART Service UUID (not required here since we connect by name)
UART_SERVICE_UUID: str = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"


# ------------------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------------------

async def main() -> None:
    """
    Description:
        Scan for the HID Proxy device by BLE name, connect to it, and remain
        connected until the user stops the script with Ctrl+C.

    Args:
        None

    Returns:
        None
    """
    print(f"Scanning for '{DEVICE_NAME}'...")

    # Attempt to find the device by name (or advertisement local_name).
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: (d.name == DEVICE_NAME) or (ad and ad.local_name == DEVICE_NAME),
        timeout=20.0,
    )

    if not device:
        print(f"Device '{DEVICE_NAME}' not found. Make sure it is advertising and nearby.")
        return

    print(f"Found device: {device}")

    # Connect to the device using BleakClient context manager.
    async with BleakClient(device) as client:
        if client.is_connected:
            print(f"Connected to {DEVICE_NAME}")
        else:
            print("Failed to connect")
            return

        # Keep the connection alive until interrupted.
        print("Press Ctrl+C to exit...")
        try:
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print("\nDisconnecting...")


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
