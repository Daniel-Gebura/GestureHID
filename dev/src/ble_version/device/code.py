####################################################################################################
# code.py
#
# Description: CircuitPython entrypoint for the Adafruit nRF52840 USB Stick. This file intentionally 
# stays very small. CircuitPython automatically executes `code.py` on boot / reload.
#
# Package layout expected on CIRCUITPY:
#
#   CIRCUITPY/
#       code.py
#       ble_receiver/
#           __init__.py
#           app.py
#           ble_io.py
#           config.py
#           hid_controller.py
#           keymaps.py
#           protocol.py
#           security.py
#       lib/
#           adafruit_ble/
#           adafruit_hid/       
#
# Author: Daniel Gebura
####################################################################################################

from ble_receiver.app import run


# Execute the BLE -> USB HID application forever.
run()