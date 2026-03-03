################################################################
# ble_communication/models.py
#
# Description:
#   Lightweight data models used by the API.
#
################################################################

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeviceTarget:
    """
    Description:
        Identifies which BLE device to connect to.

    Fields:
        name: BLE advertising name (e.g., "HID-Proxy-Control")
        address: Optional BLE address/identifier. If provided, prefer this over name.
    """
    name: str
    address: Optional[str] = None


@dataclass(frozen=True)
class Ack:
    """
    Description:
        Parsed acknowledgement from the device.

    Expected device payloads:
        {"ok": true}
        {"ok": true, "authed": true}
        {"ok": false, "error": "..."}

    Fields:
        ok: Whether command succeeded
        error: Optional error string if ok is False
        raw: The full JSON dict returned by device
    """
    ok: bool
    error: Optional[str]
    raw: dict