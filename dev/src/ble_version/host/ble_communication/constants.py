################################################################
# ble_communication/constants.py
#
# Description:
#   Centralized constants for the BLE HID Proxy host-side API.
#
################################################################

# Nordic UART Service (NUS) UUIDs (standard)
NUS_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # Write to peripheral (device RX)
NUS_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # Notify from peripheral (device TX)

# Safety: keep BLE payloads small (device also caps at JSON_MAX_LEN)
DEFAULT_MAX_JSON_BYTES = 512