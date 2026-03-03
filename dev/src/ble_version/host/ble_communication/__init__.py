################################################################
# ble_communication package
#
# Public API surface:
# - BleHidProxyClient
# - BleHidProxyConfig
# - DeviceTarget
#
################################################################

from .client import BleHidProxyClient, BleHidProxyConfig
from .models import DeviceTarget, Ack
from .exceptions import (
    BleHidProxyError,
    DeviceNotFoundError,
    ConnectionError,
    AuthenticationError,
    AckTimeoutError,
    DeviceErrorResponse,
)

__all__ = [
    "BleHidProxyClient",
    "BleHidProxyConfig",
    "DeviceTarget",
    "Ack",
    "BleHidProxyError",
    "DeviceNotFoundError",
    "ConnectionError",
    "AuthenticationError",
    "AckTimeoutError",
    "DeviceErrorResponse",
]