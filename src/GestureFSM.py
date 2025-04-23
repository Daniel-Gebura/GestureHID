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
    open_hand -> closed_fist -> open_hand -> closed_fist -> open_hand -> closed_fist -> open_hand
    within a 2-second window to toggle HID control on or off.
    """
    def __init__(self):
        self._reset()

    def _reset(self):
        """
        Reset the FSM to its initial state.
        """
        self.sequence_step = 0
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

        # Reset if more than 2 seconds have passed since last valid gesture
        if self.last_update_time is not None and (curr_time - self.last_update_time > 2.0):
            self._reset()

        # Determine what gesture is expected next
        expected_gesture = "open_hand" if self.sequence_step % 2 == 0 else "closed_fist"

        # If expected gesture is seen, increment sequence
        if gesture == expected_gesture:
            self.sequence_step += 1
            self.last_update_time = curr_time

            # If full sequence is completed, return True
            if self.sequence_step == 7:
                self._reset()
                return True
        # Reset if an invalid gesture interrupts the sequence
        elif gesture not in ["open_hand", "closed_fist", "None"]:
            self._reset()

        return False

class MouseFSM:
    """
    FSM to manage persistent mouse button press/release based on gesture.
    Uses a filtering state to suppress gesture jitter by requiring gestures to persist for 2 frames.
    """
    STATE_NONE = 0          # No mouse button is being held
    STATE_FILTERING = 1     # A new gesture has been seen once; waiting for confirmation
    STATE_HOLDING = 2       # A valid gesture is currently being held

    def __init__(self, sensitivity=1000):
        self.state = self.STATE_NONE                # Start in the no-hold state
        self.prev_gesture = None                    # Stores the previous confirmed gesture
        self.hid = HIDController()                  # HID interface for sending mouse events
        self.sensitivity = sensitivity              # Scaling factor for motion
        self.curr_buttons = 0x00                    # Currently pressed button mask
        self.filter_candidate = None                # Gesture being filtered (not yet confirmed)

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
        Update FSM based on the current gesture and send mouse input.
        Includes a jitter filter using FSM states.

        Args:
            gesture (str): Detected gesture for this frame.
            prev_xy (tuple): Previous hand position.
            curr_xy (tuple): Current hand position.
        """
        # --- STATE_NONE: Nothing is being held ---
        if self.state == self.STATE_NONE:
            if gesture in MOUSE_MAP:
                # Start filtering a potential new gesture
                self.state = self.STATE_FILTERING
                self.filter_candidate = gesture

        # --- STATE_FILTERING: Waiting for gesture to persist ---
        elif self.state == self.STATE_FILTERING:
            if gesture == self.filter_candidate:
                # Gesture confirmed as stable; press button and hold
                self.curr_buttons = MOUSE_MAP[gesture]
                self.hid.press_mouse(self.curr_buttons)
                self.state = self.STATE_HOLDING
                self.prev_gesture = gesture
                self.filter_candidate = None
            elif gesture in MOUSE_MAP:
                # Switch filter candidate if a different pressable gesture appears
                self.filter_candidate = gesture
            else:
                # Gesture invalid or open; cancel filtering
                self.state = self.STATE_NONE
                self.filter_candidate = None

        # --- STATE_HOLDING: A button is currently being held ---
        elif self.state == self.STATE_HOLDING:
            if gesture == "open_hand":
                # Open hand releases button
                self.hid.release_mouse()
                self.state = self.STATE_NONE
                self.curr_buttons = 0x00
                self.prev_gesture = None
            elif gesture != self.prev_gesture:
                # Gesture changed; release current and filter new candidate if valid
                self.hid.release_mouse()
                if gesture in MOUSE_MAP:
                    self.state = self.STATE_FILTERING
                    self.filter_candidate = gesture
                else:
                    # Invalid new gesture; return to idle
                    self.state = self.STATE_NONE
                    self.curr_buttons = 0x00
                    self.prev_gesture = None

        # Calculate and apply motion (allowed even during filtering or holding)
        dx, dy = self._calculate_mouse_movement(prev_xy, curr_xy)
        if gesture != "closed_fist":
            self._move_mouse(dx, dy)

class KeyboardFSM:
    """
    FSM to manage persistent key press/release behavior for gestures.
    ESC key is handled as a tap-only gesture.
    Uses a filtering state to suppress gesture jitter by requiring gestures to persist for 2 frames.
    """
    STATE_NONE = 0          # No key is being held
    STATE_FILTERING = 1     # A gesture appeared once; waiting for confirmation
    STATE_HOLDING = 2       # A key is currently being held

    def __init__(self):
        self.state = self.STATE_NONE                 # Initial state
        self.prev_gesture = None                    # Last confirmed gesture
        self.last_was_esc = False                   # Flag to handle ESC key taps
        self.hid = HIDController()                  # HID interface for sending keyboard input
        self.filter_candidate = None                # Gesture seen once, waiting for confirmation

    def update(self, gesture):
        """
        Update FSM based on current gesture and send keyboard input.

        Args:
            gesture (str): Detected gesture for this frame.
        """
        # --- STATE_NONE: No key currently pressed ---
        if self.state == self.STATE_NONE:
            if gesture in KEY_MAP:
                # Begin filtering this gesture
                self.state = self.STATE_FILTERING
                self.filter_candidate = gesture

        # --- STATE_FILTERING: Waiting for gesture to stabilize ---
        elif self.state == self.STATE_FILTERING:
            if gesture == self.filter_candidate:
                # Confirmed stable; press corresponding key
                key = KEY_MAP[gesture]
                self.hid.press_key(key)
                self.prev_gesture = gesture
                self.state = self.STATE_HOLDING
                self.filter_candidate = None
            elif gesture in KEY_MAP:
                # Update filter to new candidate
                self.filter_candidate = gesture
            else:
                # Invalid gesture; cancel filtering
                self.state = self.STATE_NONE
                self.filter_candidate = None

        # --- STATE_HOLDING: Key is currently being held ---
        elif self.state == self.STATE_HOLDING:
            if gesture == "open_hand":
                # Open hand releases key
                self.hid.release_keys()
                self.prev_gesture = None
                self.state = self.STATE_NONE
            elif gesture != self.prev_gesture:
                # Gesture changed; release current and filter new candidate if valid
                self.hid.release_keys()
                if gesture in KEY_MAP:
                    self.state = self.STATE_FILTERING
                    self.filter_candidate = gesture
                else:
                    # Invalid gesture; go back to idle
                    self.state = self.STATE_NONE
                    self.prev_gesture = None
                    self.filter_candidate = None