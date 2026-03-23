####################################################################################################
# ble_receiver/app.py
#
# Description: Main application loop for the CircuitPython BLE -> USB HID receiver.
#
# High-level runtime flow:
#   1) Advertise Nordic UART BLE service.
#   2) Accept connection from central.
#   3) Enforce connection security.
#   4) Read newline-delimited JSON commands over BLE UART.
#   5) Require session authorization before HID activity.
#   6) Route commands to keyboard or mouse actions.
#   7) Send structured acknowledgments / errors back to the sender.
#   8) On disconnect, release all HID state to avoid stuck keys/buttons.
#
# Author: Daniel Gebura
####################################################################################################

import time

from ble_receiver.ble_io import BLEUARTServer
from ble_receiver.config import IDLE_SLEEP_S
from ble_receiver.hid_controller import HIDController
from ble_receiver.protocol import JSONLineProtocol
from ble_receiver.protocol import ProtocolError
from ble_receiver.security import SecurityManager


class BLEHIDReceiverApp:
    """
    Description:
        Long-running application object that coordinates BLE transport, protocol parsing,
        security enforcement, and USB HID output.
    """

    def __init__(self):
        """
        Description:
            Construct the subsystem objects used by the runtime.

        Returns:
            None
        """
        self.ble_server = BLEUARTServer()
        self.hid = HIDController()
        self.protocol = JSONLineProtocol()
        self.security = SecurityManager()

    def _safe_write_response(self, payload_bytes):
        """
        Description:
            Best-effort UART response write.
            BLE notification / UART writes can fail transiently if the connection changes state at an
            unlucky time. A failed response should not crash the device runtime.

        Returns:
            None
        """
        try:
            self.ble_server.write_json_bytes(payload_bytes)
        except Exception:
            pass

    def _handle_disconnect_transition(self):
        """
        Description:
            Detect BLE disconnect transitions and release all HID state to avoid stuck keys/buttons.

        Returns:
            None
        """
        # Check if the ble server is still c9nnected
        currently_connected = self.ble_server.is_connected()

        # If we previously were connected on last check and now we are not = Disconnect transition
        if self.security.was_connected and not currently_connected:
            self.hid.release_all()  # Release all HID keys/buttons
            self.security.session_authenticated = False  # Clear the authenticated session flag

        # Update the previous connection state for the next check
        self.security.was_connected = currently_connected

    def _wait_for_secured_connection(self):
        """
        Description:
            Ensure the device is advertising, wait for a connection, and enforce security requirements.
            
        Returns:
            True when a valid secured connection is ready.
            False when the connection was rejected and the app should restart the advertisement loop.
        """
        # 1. Check if the device is connected.
        if self.ble_server.is_connected():
            print("Already connected")
            return True

        # 2. If not connected, reset security state.
        self.security.reset_session_authentication()

        # 3. Start advertising, sleep until a central connects, then stop advertising.
        try:
            self.ble_server.start_advertising()
            print("Advertising started, waiting for connection...")
            while not self.ble_server.is_connected():
                time.sleep(IDLE_SLEEP_S)
            print("Central connected, stopping advertising.")
            self.ble_server.stop_advertising()
        except Exception:
            # Ignore benign BLE stack state errors.
            print("Error during advertising or connection, restarting advertisement loop.")
            return False

        # 4. Check if the connected central meets security requirenments.
        connection = self.ble_server.get_active_connection()
        if connection is None:
            print("No active connection found after connection event, restarting advertisement loop.")
            return False
        if not self.security.enforce_connection_security(connection):
            print("Connection failed security requirements, restarting advertisement loop.")
            return False

        # 5. If we reach this point, the connection is valid.
        print("Central Connected")
        return True
    
    def _process_one_command(self, raw_line_bytes):
        """
        Description:
            Parse, authenticate, execute, and acknowledge one inbound command payload.

        Args: 
            raw_line_bytes:
                Raw bytes returned by UARTService.readline().

        Returns:
            None
        """
        # 1. Parse the raw bytes into a command dictionary.
        try:
            cmd = self.protocol.parse_raw_line(raw_line_bytes)
        except ProtocolError as exc:
            self._safe_write_response(self.protocol.build_error_response(str(exc)))
            return

        try:
            # 2. Check if the command contains valid authentication.
            if not self.security.handle_auth_command(cmd):
                self._safe_write_response(self.protocol.build_error_response("unauthorized"))
                return

            # 3. If this is an explicit auth command, send the auth OK response and return early.
            if cmd.get("type") == "auth":
                self._safe_write_response(self.protocol.build_auth_ok_response())
                return

            # 4. Route the command to the appropriate HID action.
            self.hid.route_command(cmd)
            self._safe_write_response(self.protocol.build_ok_response())

        except Exception as exc:
            self._safe_write_response(self.protocol.build_error_response(str(exc)))

    def run_forever(self):
        """
        Description:
            Execute the main application loop forver.
        """
        while True:
            # 1. Check for disconnect transition.
            self._handle_disconnect_transition()

            # 2. Wait for a secured connection.
            if not self._wait_for_secured_connection():
                continue

            # 3. Once connected, just read and process commands until disconnect.
            while self.ble_server.is_connected():
                raw_line_bytes = self.ble_server.read_line()

                if not raw_line_bytes:
                    time.sleep(IDLE_SLEEP_S)
                    continue

                self._process_one_command(raw_line_bytes)

def run():
    """
    Description:
        Construct the application object and run the main loop forever.
    """
    app = BLEHIDReceiverApp()
    app.run_forever()