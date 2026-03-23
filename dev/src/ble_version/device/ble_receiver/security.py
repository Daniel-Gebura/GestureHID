####################################################################################################
# ble_receiver/security.py
#
# Description: Connection security and session authorization state.
#
# This module manages three related concerns:
#   1) BLE pairing / bonding enforcement.
#   2) Allow-listing the first central address seen after reset.
#   3) Lightweight per-session token authorization for incoming commands.
#
# Author: Daniel Gebura
####################################################################################################

from ble_receiver.config import AUTH_TOKEN
from ble_receiver.config import REQUIRE_BONDING

class SecurityManager:
    """
    Description:
        Holds mutable connection/session security state for the running application.
    """

    def __init__(self):
        """
        Description: 
            Initialize security state.
            
        State fields:
            allowed_central_address:
                Once  the first accpetable central connects, its address is cached until reset.
            session_authenticated:
                Cleared on disconnect. Set to True after a valid auth token is received.
            was_connected:
                Tracks the previous BLE connection state so the app can disconnect transitions.
        """
        self.allowed_central_address = None
        self.session_authenticated = False
        self.was_connected = False

    @staticmethod
    def _format_address_hex(address_object):
        """
        Description:
            Convert a BLE address-like object into a stable uppercase string.

        Args:
            address_object: A BLE address-like object that can be converted to a string or bytes.

        Returns:
            Example format: "AA:BB:CC:DD:EE:FF"
            Returns None when the address cannot be determined.
        """
        try:
            address_as_string = str(address_object)
            if address_as_string and ":" in address_as_string:
                return address_as_string.upper()
        except Exception:
            pass

        try:
            address_bytes = bytes(address_object)
            if address_bytes and len(address_bytes) == 6:
                return ":".join("{:02X}".format(byte_value) for byte_value in address_bytes[::-1])
        except Exception:
            pass

        return None

    def get_connection_address_str(self, connection):
        """
        Description:
            Extract the peer central BLE address from a CircuitPython BLE connection object.

        Args:
            connection: A CircuitPython BLE connection object.

        Returns:
            The peer central's BLE address as a string in "AA:BB:CC:DD:EE:FF" format, or None if it cannot be determined.
        """
        try:
            bleio_connection = getattr(connection, "_bleio_connection", None)
            if bleio_connection is not None:
                address_object = getattr(bleio_connection, "address", None)
                if address_object is not None:
                    return self._format_address_hex(address_object)
        except Exception:
            pass

        return None

    def reset_session_authentication(self):
        """
        Description:
            Clear per-connection authorization state.

        """
        self.session_authenticated = False

    def note_connection_state(self, is_connected):
        """
        Description:
            Update the cached previous connection state.
        """
        self.was_connected = is_connected
    
    def handle_auth_command(self, cmd):
        """
        Description:
            Apply lightweight session authentication rules.

        Supported patterns:
            1) Explicit auth command:
                {"type":"auth", "token":"CHANGE_ME"}

            2) Inline auth field on a normal command before session is authenticated:
                {"type":"mouse", "action":"move", "dx":10, "auth":"CHANGE_ME"}

        Args:
            cmd: A dictionary representing the parsed command payload.

        Returns:
            True if the session is now authenticated.
            False if the command did not provide valid auth.
        """
        if self.session_authenticated:
            return True

        if cmd.get("type") == "auth":
            token = cmd.get("token", None)
            if isinstance(token, str) and token == AUTH_TOKEN:
                self.session_authenticated = True
                return True
            return False

        token = cmd.get("auth", None)
        if isinstance(token, str) and token == AUTH_TOKEN:
            self.session_authenticated = True
            return True

        return False

    def enforce_connection_security(self, connection):
        """"
        Description:
            Enforce BLE pairing/bonding and central allow-list behavior.
            
        Args: 
            connection: A BLS connection object that is trying to become valid.

        Returns: 
            True if connection is accepted.
            False if connection is rejected.
        """
        if REQUIRE_BONDING:
            try:
                if not connection.paired:
                    connection.pair(bond=True)
            except Exception:
                try:
                    connection.disconnect()
                except Exception:
                    pass
                return False

            try:
                if not connection.paired:
                    connection.disconnect()
                return bool(connection.paired)
            except Exception:
                try:
                    connection.disconnect()
                except Exception:
                    pass
                return False

        peer_address = self.get_connection_address_str(connection)

        # If address discovery fails, fail closed.
        if peer_address is None:
            try:
                connection.disconnect()
            except Exception:
                pass
            return False

        if self.allowed_central_address is None:
            self.allowed_central_address = peer_address
            return True

        if peer_address != self.allowed_central_address:
            try:
                connection.disconnect()
            except Exception:
                pass
            return False

        return True