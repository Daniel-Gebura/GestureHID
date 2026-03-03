################################################################
# ble_communication/exceptions.py
#
# Description:
#   Typed exceptions so your top-level program can handle failures cleanly.
#
################################################################

class BleHidProxyError(Exception):
    """Base class for all BLE HID proxy errors."""


class DeviceNotFoundError(BleHidProxyError):
    """Raised when the BLE device cannot be discovered by name/address."""


class ConnectionError(BleHidProxyError):
    """Raised when BLE connect/disconnect operations fail."""


class AuthenticationError(BleHidProxyError):
    """Raised when auth handshake fails."""


class AckTimeoutError(BleHidProxyError):
    """Raised when waiting for device ack times out."""


class DeviceErrorResponse(BleHidProxyError):
    """Raised when device responds with ok=false."""

    def __init__(self, error_message: str, payload: dict | None = None) -> None:
        super().__init__(error_message)
        self.error_message = error_message
        self.payload = payload or {}