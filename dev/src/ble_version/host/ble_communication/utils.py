################################################################
# ble_communication/utils.py
#
# Description:
#   Small utilities (no BLE logic here).
#
################################################################

import json


def json_dumps_compact(obj: dict) -> str:
    """
    Description:
        Compact JSON serializer to reduce BLE payload size.

    Returns:
        JSON string without extra spaces.
    """
    return json.dumps(obj, separators=(",", ":"))


def ensure_trailing_newline(s: str) -> str:
    """
    Description:
        Ensure newline termination for line-based UART framing.
    """
    if s.endswith("\n"):
        return s
    return s + "\n"