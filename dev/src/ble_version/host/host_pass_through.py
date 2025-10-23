# ==============================================================================
# host_send_keys.py
# Connects to a BLE device by name and forwards each key you press to the
# device over Nordic UART Service (NUS), one byte per keypress.
# Author: Daniel
# Date: 2025-09-30
# ==============================================================================

import asyncio
import sys
from contextlib import contextmanager
from typing import Iterator, Optional

from bleak import BleakClient, BleakScanner


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

DEVICE_NAME: str = "HID-Proxy-Simple"

# Nordic UART Service UUIDs (RX = host→peripheral | TX = peripheral→host)
UART_RX_UUID: str = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # write to peripheral
UART_TX_UUID: str = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # notify from peripheral (unused)


# ------------------------------------------------------------------------------
# Key capture (cross-platform)
# ------------------------------------------------------------------------------

@contextmanager
def raw_mode() -> Iterator[None]:
    """
    Description:
        Put stdin into raw, non-echo mode on POSIX so we can read a single
        character at a time without waiting for Enter. No-op on Windows.

    Args:
        None

    Returns:
        None
    """
    # On Windows, console handling is different; do nothing.
    if sys.platform.startswith("win"):
        yield
        return

    # On POSIX, switch terminal to raw mode and ensure we restore it.
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def get_one_key() -> Optional[str]:
    """
    Description:
        Read a single Unicode character from the keyboard without requiring
        Enter. Uses `msvcrt.getwch()` on Windows and raw stdin on POSIX.

    Args:
        None

    Returns:
        A single-character string if available; otherwise None.
    """
    if sys.platform.startswith("win"):
        # On Windows, read a wide char; returns str of length 1.
        import msvcrt

        if msvcrt.kbhit():
            return msvcrt.getwch()
        return None
    else:
        # On POSIX, stdin is already in raw mode (via raw_mode()).
        ch = sys.stdin.read(1)
        return ch if ch else None


# ------------------------------------------------------------------------------
# BLE helpers
# ------------------------------------------------------------------------------

async def find_device_by_name(name: str, timeout: float = 10.0):
    """
    Description:
        Scan for a BLE device that advertises with the provided name.

    Args:
        name: Target BLE advertising device name.
        timeout: Scan duration in seconds.

    Returns:
        The discovered device object (platform-specific) or None if not found.
    """
    print(f"Scanning for '{name}' …")
    device = await BleakScanner.find_device_by_filter(
        # Match either the device name or advertisement's local name.
        lambda d, ad: (d.name == name) or (ad and ad.local_name == name),
        timeout=timeout,
    )
    return device


# ------------------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------------------

async def main() -> None:
    """
    Description:
        Connect to the target BLE device and forward each key you press to the
        device over NUS, one byte per keypress. ASCII only in this minimal
        example (printable 32..126 recommended). Press Ctrl+C to quit.

    Args:
        None

    Returns:
        None
    """
    # Discover the device by its advertised name.
    device = await find_device_by_name(DEVICE_NAME, timeout=25.0)
    if not device:
        print(f"Device '{DEVICE_NAME}' not found. Is it advertising and nearby?")
        return

    print(f"Found: {device}. Connecting…")

    # Use BleakClient context manager to connect and clean up automatically.
    async with BleakClient(device) as client:
        if not client.is_connected:
            print("Failed to connect.")
            return

        print(f"Connected to {DEVICE_NAME}. Type keys; Ctrl+C to quit.")

        # Optional: enable notifications for device ACK/debug (not used by the
        # minimal device code; uncomment if you add TX notifications).
        # await client.start_notify(
        #     UART_TX_UUID,
        #     lambda handle, data: print("DEVICE:", data.decode(errors="ignore").strip())
        # )

        try:
            # Put terminal in raw mode on POSIX so we get characters immediately.
            with raw_mode():
                while True:
                    ch = get_one_key()
                    if not ch:
                        # Nothing to send yet; allow the event loop to breathe.
                        await asyncio.sleep(0.001)
                        continue

                    # Minimal pass-through: take the first UTF-8 byte (ASCII).
                    b = ch.encode("utf-8", errors="ignore")[:1]
                    if not b:
                        continue

                    # Write the single byte to the device's RX characteristic.
                    await client.write_gatt_char(UART_RX_UUID, b)

                    # Optional local echo so the user sees what was sent.
                    sys.stdout.write(ch)
                    sys.stdout.flush()

        except KeyboardInterrupt:
            # Graceful shutdown on Ctrl+C.
            print("\nExiting…")
        finally:
            # If you enabled notifications above, remember to stop them here.
            # try:
            #     await client.stop_notify(UART_TX_UUID)
            # except Exception:
            #     pass
            pass


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
