################################################################
# GestureFSM.py
#
# Description: Defines finite state machines (FSMs) for controlling HID behavior
# in response to gesture sequences. Abstracts interaction with HIDController
# to decouple control logic from the main gesture processing script.
#
# Author: Daniel Gebura
################################################################

import time
from hid_controller import HIDController, MOUSE_LEFT, MOUSE_RIGHT, KEYCODES

# Constants ---------------------------------------------------------

# Maps gesture names to key values
KEY_MAP = {
    "forward_point": "w",
    "back_point": "s",
    "left_point": "a",
    "right_point": "d",
    "index_thumb": "esc"
}

# Maps gesture names to mouse buttons
MOUSE_MAP = {
    "index_thumb": MOUSE_LEFT,
    "pinky_thumb": MOUSE_RIGHT
}

# Finite State Machines ---------------------------------------------

class HIDToggleFSM:
    """
    FSM to recognize a sequence of gestures:
    open_hand -> closed_fist -> open_hand -> closed_fist -> open_hand
    within a 3-second window to toggle HID control on or off.
    """
    # FSM State Names
    STATE_IDLE = 0
    STATE_FIRST_OPEN = 1
    STATE_FIRST_FIST = 2
    STATE_SECOND_OPEN = 3
    STATE_SECOND_FIST = 4

    def __init__(self):
        self.state = STATE_IDLE
        self.last_update_time = None

    def update(self, gesture):
        """
        Update FSM state based on gesture.

        Args:
            gesture (str): Current right-hand gesture.

        Returns:
            bool: True if the toggle sequence is completed.
        """
        # Track time for possible timeout
        curr_time = time.time()

        # Reset the state if timeout occurs
        if (self.last_update_time is not None) and (curr_time - self.last_update_time > 3.0):
            self.state = STATE_IDLE

        # Match the current state and update based on gesture
        match self.state:
            case self.STATE_IDLE:
                # Only leave idle state if the first gesture is open_hand
                if gesture == "open_hand":
                    # Start the sequence
                    self.state = STATE_FIRST_OPEN
                    self.last_update_time = curr_time
            case self.STATE_FIRST_OPEN:
                # Transition to next state if the gesture is closed_fist
                if gesture == "closed_fist":
                    self.state = STATE_FIRST_FIST
            case self.STATE_FIRST_FIST:
                # Transition to next state if the gesture is open_hand
                if gesture == "open_hand":
                    self.state = STATE_SECOND_OPEN
            case self.STATE_SECOND_OPEN:
                # Transition to next state if the gesture is closed_fist
                if gesture == "closed_fist":
                    self.state = STATE_SECOND_FIST
            case self.STATE_SECOND_FIST:
                # Complete the sequence if the gesture is open_hand
                if gesture == "open_hand":
                    self.state = STATE_IDLE  # Reset the state
                    return True  # Toggle the HID state
            case _:
                # Reset the state if an unexpected, known gesture is detected
                if gesture not in ["open_hand", "closed_fist", "None"]:
                    self.state = STATE_IDLE

        return False

class MouseFSM:
    """
    FSM to manage persistent mouse button press/release based on gesture.
    """
    # FSM State Names
    STATE_NONE = 0
    STATE_HOLDING = 1

    def __init__(self):
        self.state = self.STATE_NONE
        self.prev_gesture = None
        self.hid = HIDController()
        self.curr_buttons = 0x00  # Bitmask of current buttons being held

    def update(self, gesture):
        """
        Update the mouse button press/release logic.

        Args:
            gesture (str): Current right-hand gesture.
        """
        # Match the current state and update based on gesture
        match self.state:
            case self.STATE_NONE:
                # Begin pressing the corresponding mouse button if gesture is recognized
                if gesture in MOUSE_MAP:
                    self.curr_buttons = MOUSE_MAP[gesture]  # Save the current button bitmask
                    self.hid.press_mouse(self.curr_buttons)  # Press the corresponding button
                    self.state = self.STATE_HOLDING  # Transition to the holding state
                    self.prev_gesture = gesture  # Save this gesture for later comparison
            case self.STATE_HOLDING:
                # Release the mouse button if the gesture is open_hand
                if gesture == "open_hand":
                    self.hid.release_mouse()  # Release the mouse button
                    self.state = self.STATE_NONE  # Transition back to the none state
                    self.curr_buttons = 0x00  # Reset the current button bitmask
                    self.prev_gesture = None  # Reset the previous gesture
                # Release the mouse button if the gesture changes
                elif gesture != self.prev_gesture:
                    self.hid.release_mouse()  # Release the mouse button
                    # If the new gesture is recognized, press the new button
                    if gesture in MOUSE_MAP:
                        self.curr_buttons = MOUSE_MAP[gesture]  # Save the current button bitmask
                        self.hid.press_mouse(self.curr_buttons)  # Press the corresponding button
                        self.prev_gesture = gesture  # Save this gesture for later comparison
                    # If the new gesture is not recognized, reset the state
                    else:
                        self.state = self.STATE_NONE  # Transition back to the none state
                        self.curr_buttons = 0x00  # Reset the current button bitmask
                        self.prev_gesture = None  # Reset the previous gesture
                # Else, continue pressing the mouse button

    def move_mouse(self, dx, dy):
        """
        Move the mouse while preserving the current button state.

        Args:
            dx (float): Pixels to move in X direction.
            dy (float): Pixels to move in Y direction.
        """
        self.hid.move_mouse(int(dx), int(dy), buttons=self.curr_buttons)

class KeyboardFSM:
    """
    FSM to manage persistent key press/release behavior for gestures.
    ESC key is handled as a tap-only gesture.
    """
    # FSM State Names
    STATE_NONE = 0
    STATE_HOLDING = 1

    def __init__(self):
        self.state = self.STATE_NONE
        self.prev_gesture = None
        self.last_was_esc = False
        self.hid = HIDController()

    def update(self, gesture):
        """
        Update the keyboard key logic.

        Args:
            gesture (str): Current left-hand gesture.
        """
        # Match the current state and update based on gesture
        match self.state:
            case self.STATE_NONE:
                # Begin pressing the corresponding key if gesture is recognized
                if gesture in KEY_MAP:
                    key = KEY_MAP[gesture]  # Get the key value
                    self.hid.press_key(key)  # Begin holding the key
                    self.prev_gesture = gesture  # Save this gesture for later comparison
                    self.state = self.STATE_HOLDING  # Transition to the holding state
            case self.STATE_HOLDING:
                # Release the key if the gesture is open_hand
                if gesture == "open_hand":
                    self.hid.release_keys()
                    self.prev_gesture = None
                    self.state = self.STATE_NONE
                # Release the key if the gesture changes
                elif gesture != self.prev_gesture:
                    self.hid.release_keys()
                    # If the new gesture is recognized, press the new key
                    if gesture in KEY_MAP:
                        key = KEY_MAP[gesture]
                        self.hid.press_key(key)
                        self.prev_gesture = gesture
                    else:
                    # If the new gesture is not recognized, reset the state
                        self.state = self.STATE_NONE
                        self.prev_gesture = None
                # Else, continue pressing the key