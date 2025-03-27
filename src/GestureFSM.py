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
from HIDController import HIDController, MOUSE_LEFT, MOUSE_RIGHT, KEYCODES

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
    within a 2-second window to toggle HID control on or off.
    """
    # FSM State Names
    STATE_IDLE = 0
    STATE_FIRST_OPEN = 1
    STATE_FIRST_FIST = 2
    STATE_SECOND_OPEN = 3
    STATE_SECOND_FIST = 4

    def __init__(self):
        self.state = self.STATE_IDLE
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
        if (self.last_update_time is not None) and (curr_time - self.last_update_time > 2.0):
            self.state = self.STATE_IDLE

        # Match the current state and update based on gesture
        if self.state == self.STATE_IDLE:
            if gesture == "open_hand":
                self.state = self.STATE_FIRST_OPEN
                self.last_update_time = curr_time
        elif self.state == self.STATE_FIRST_OPEN:
            if gesture == "closed_fist":
                self.state = self.STATE_FIRST_FIST
        elif self.state == self.STATE_FIRST_FIST:
            if gesture == "open_hand":
                self.state = self.STATE_SECOND_OPEN
        elif self.state == self.STATE_SECOND_OPEN:
            if gesture == "closed_fist":
                self.state = self.STATE_SECOND_FIST
        elif self.state == self.STATE_SECOND_FIST:
            if gesture == "open_hand":
                self.state = self.STATE_IDLE
                return True
        else:
            if gesture not in ["open_hand", "closed_fist", "None"]:
                self.state = self.STATE_IDLE

        return False

class MouseFSM:
    """
    FSM to manage persistent mouse button press/release based on gesture.
    """
    # FSM State Names
    STATE_NONE = 0
    STATE_HOLDING = 1

    def __init__(self, sensitivity=1000):
        self.state = self.STATE_NONE
        self.prev_gesture = None
        self.hid = HIDController()
        self.sensitivity = sensitivity
        self.curr_buttons = 0x00

    def _calculate_mouse_movement(self, prev_xy, curr_xy):
        """
        Calculate the change in mouse x and y coordinates based on the previous and current normalized coordinates.

        Args:
            prev_xy (tuple or None): (prev_x, prev_y) if available, otherwise None.
            curr_xy (tuple): (curr_x, curr_y)

        Returns:
            tuple: (delta_x, delta_y) computed as pixel differences.
        """
        if prev_xy is None or curr_xy is None:
            return 0, 0
        prev_x, prev_y = prev_xy
        curr_x, curr_y = curr_xy
        delta_x = (curr_x - prev_x) * self.sensitivity
        delta_y = (curr_y - prev_y) * self.sensitivity
        return delta_x, delta_y

    def _move_mouse(self, dx, dy):
        """
        Move the mouse while preserving the current button state.

        Args:
            dx (float): Pixels to move in X direction.
            dy (float): Pixels to move in Y direction.
        """
        self.hid.move_mouse(int(dx), int(dy), buttons=self.curr_buttons)

    def update(self, gesture, prev_xy, curr_xy):
        """
        Update the mouse button press/release logic.

        Args:
            gesture (str): Current right-hand gesture.
        """
        if self.state == self.STATE_NONE:
            if gesture in MOUSE_MAP:
                self.curr_buttons = MOUSE_MAP[gesture]
                self.hid.press_mouse(self.curr_buttons)
                self.state = self.STATE_HOLDING
                self.prev_gesture = gesture
        elif self.state == self.STATE_HOLDING:
            if gesture == "open_hand":
                self.hid.release_mouse()
                self.state = self.STATE_NONE
                self.curr_buttons = 0x00
                self.prev_gesture = None
            elif gesture != self.prev_gesture:
                self.hid.release_mouse()
                if gesture in MOUSE_MAP:
                    self.curr_buttons = MOUSE_MAP[gesture]
                    self.hid.press_mouse(self.curr_buttons)
                    self.prev_gesture = gesture
                else:
                    self.state = self.STATE_NONE
                    self.curr_buttons = 0x00
                    self.prev_gesture = None

        dx, dy = self._calculate_mouse_movement(prev_xy, curr_xy)
        if gesture != "closed_fist":
            self._move_mouse(dx, dy)

class KeyboardFSM:
    """
    FSM to manage persistent key press/release behavior for gestures.
    ESC key is handled as a tap-only gesture.
    """
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
        if self.state == self.STATE_NONE:
            if gesture in KEY_MAP:
                key = KEY_MAP[gesture]
                self.hid.press_key(key)
                self.prev_gesture = gesture
                self.state = self.STATE_HOLDING
        elif self.state == self.STATE_HOLDING:
            if gesture == "open_hand":
                self.hid.release_keys()
                self.prev_gesture = None
                self.state = self.STATE_NONE
            elif gesture != self.prev_gesture:
                self.hid.release_keys()
                if gesture in KEY_MAP:
                    key = KEY_MAP[gesture]
                    self.hid.press_key(key)
                    self.prev_gesture = gesture
                else:
                    self.state = self.STATE_NONE
                    self.prev_gesture = None