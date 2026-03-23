####################################################################################################
# ble_receiver/config.py
#
# Description: Centralized configuration values for the BLE receiver application.
#
# Aurthor: Daniel Gebura
####################################################################################################

# --------------------------------------------------------------------------------------------------
# BLE / Protocol Configuration
# --------------------------------------------------------------------------------------------------

# BLE advertising name visible to scanning central devices.
DEVICE_NAME = "HID-Proxy-Control"

# Maximum accepted newline-delimited JSON command length in bytes.
# This protects the device from oversized payloads that waste memory or indicate a malformed sender.
JSON_MAX_LEN = 512

# Idle sleep time used while polling BLE state and UART input.
# Small values improve responsiveness but cost more CPU time.
IDLE_SLEEP_S = 0.01


# --------------------------------------------------------------------------------------------------
# Mouse HID Limits
# --------------------------------------------------------------------------------------------------

# Standard mouse HID reports use signed 8-bit deltas.
# Valid range is therefore -127..127.
MOUSE_DELTA_MIN = -127
MOUSE_DELTA_MAX = 127


# --------------------------------------------------------------------------------------------------
# Connection Security Configuration
# --------------------------------------------------------------------------------------------------

# Require BLE pairing / bonding before accepting commands.
# If enabled, a connecting central must successfully pair.
REQUIRE_BONDING = True

# Lightweight application-layer shared secret.
# This is separate from BLE pairing and is used to ensure that only a known sender process issues
# commands after connection is established.
AUTH_TOKEN = "CHANGE_ME"
