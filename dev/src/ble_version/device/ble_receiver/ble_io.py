####################################################################################################
# ble_receiver/ble_io.py
#
# Description: BLE radio and Nordic UART service wrapper. This module isolates BLE setup and the 
# handful of recurring BLE utility operations used by the main application loop.
#
# Author: Daniel Gebura
####################################################################################################

from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

from ble_receiver.config import DEVICE_NAME

class BLEUARTServer:
    """
    Description:
        Wrapper around BLERadio + Nordic UART service.
    """

    def __init__(self):
        """
        Description:
            Create the BLE radio, UART service, and corresponding advertisement payload.
        """
        self.ble = BLERadio()
        self.ble.name = DEVICE_NAME
        self.uart = UARTService()
        self.uart_advertisement = ProvideServicesAdvertisement(self.uart)

    def is_connected(self):
        """
        Description:
            Return True when at least one active BLE connection exists.
        """
        return self.ble.connected

    def get_active_connection(self):
        """
        Description:
            Return the first active BLE connection, or None if unavailable.
        """
        try:
            connection_list = self.ble.connections
            if connection_list and connection_list[0] is not None and connection_list[0].connected:
                return connection_list[0]
        except Exception:
            pass

        return None

    def start_advertising(self):
        """
        Description:
            Start BLE advertising.

        Notes:
            Best-effort wrapper. Callers intentionally ignore benign stack errors such as already
            advertising or transient stack states.
        """
        self.ble.start_advertising(self.uart_advertisement)

    def stop_advertising(self):
        """
        Description:
            Stop BLE advertising.
        """
        self.ble.stop_advertising()

    def read_line(self):
        """
        Description:
            Read one newline-delimited UART payload from the connected BLE central.

        Returns:
            Bytes ending in newline when available, or None when no full line is buffered yet.
        """
        return self.uart.readline()

    def write_json_bytes(self, payload_bytes):
        """
        Description:
            Write a pre-encoded byte payload back to the BLE central.
        """
        self.uart.write(payload_bytes)