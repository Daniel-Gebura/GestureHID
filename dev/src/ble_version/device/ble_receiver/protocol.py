####################################################################################################
# ble_receiver/protocol.py
#
# Description: Protocol parsing and response formatting helpers.
#
#   The device expects newline-delimited JSON messages over BLE UART. This module is responsible for:
#       - decoding incoming bytes safely,
#       - validating high-level framing constraints,
#       - formatting success / error responses back to the sender.
#
# Author: Daniel Gebura
####################################################################################################

import json

from ble_receiver.config import JSON_MAX_LEN


class ProtocolError(Exception):
    """
    Description:
        Explicit protocol-level exception for malformed input.
    """


class JSONLineProtocol:
    """
    Description:
        Helper for parsing newline-delimited JSON commands and generating structured responses.
    """

    @staticmethod
    def parse_raw_line(raw_line_bytes):
        """
        Description:
            Parse one inbound BLE UART line into a command dictionary.

        Args:
            raw_line_bytes:
                Raw bytes returned by UARTService.readline().

        Returns:
            Parsed command dictionary.

        Raises:
            ProtocolError:
                If input is missing, oversized, invalid UTF-8, invalid JSON, or not a JSON object.
        """
        if not raw_line_bytes:
            raise ProtocolError("empty payload")

        if len(raw_line_bytes) > JSON_MAX_LEN:
            raise ProtocolError("payload too large")

        try:
            decoded_line = raw_line_bytes.decode("utf-8").strip()
        except Exception:
            raise ProtocolError("payload is not valid UTF-8")

        if not decoded_line:
            raise ProtocolError("empty payload")

        try:
            parsed_object = json.loads(decoded_line)
        except Exception:
            raise ProtocolError("payload is not valid JSON")

        if not isinstance(parsed_object, dict):
            raise ProtocolError("top-level JSON value must be an object")

        return parsed_object

    @staticmethod
    def build_ok_response():
        """
        Description:
            Build the standard success response.
        """
        return b'{"ok":true}\n'

    @staticmethod
    def build_auth_ok_response():
        """
        Description:
            Build the success response for an explicit auth command.
        """
        return b'{"ok":true,"authed":true}\n'

    @staticmethod
    def build_error_response(message):
        """
        Description:
            Build a structured JSON error response from a Python string.
        """
        return ('{"ok":false,"error":' + json.dumps(str(message)) + "}\n").encode("utf-8")
