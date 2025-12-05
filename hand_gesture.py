import cv2
import math

# ---------------------------------------------------------------------
# Try to import mediapipe.
# On Jetson this may fail or be partially broken, so we guard it.
# ---------------------------------------------------------------------
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    print("[HandGestureService] Mediapipe loaded successfully.")
except Exception as e:
    mp = None
    MP_AVAILABLE = False
    print("[HandGestureService] Mediapipe NOT available, gestures disabled.")
    print("  Reason:", e)


class HandGestureService:
    """
    Uses MediaPipe Hands to classify a hand as 'open' or 'closed'
    based on how many fingers are extended.

    On systems where Mediapipe cannot be imported (e.g. broken ARM build),
    classify() will always return None so the main app does not crash.
    """

    def __init__(self):
        self.enabled = MP_AVAILABLE

        if not MP_AVAILABLE:
            self.hands = None
            return

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def set_enabled(self, flag: bool):
        # allow external toggle, but only if mediapipe actually works
        self.enabled = flag and MP_AVAILABLE

    def classify(self, frame):
        """
        Return 'open', 'closed', or None.
        If Mediapipe is unavailable or disabled, returns None.
        """
        if not self.enabled or self.hands is None:
            return None

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            return None

        hand = result.multi_hand_landmarks[0].landmark

        # same finger logic you had before
        finger_pairs = [
            (5, 8),     # index (MCP, TIP)
            (9, 12),    # middle
            (13, 16),   # ring
            (17, 20),   # pinky
        ]

        extended_count = 0
        for mcp_idx, tip_idx in finger_pairs:
            mcp = hand[mcp_idx]
            tip = hand[tip_idx]
            if tip.y < mcp.y:
                extended_count += 1

        return "open" if extended_count >= 3 else "closed"


# Optional standalone test (camera only). Not used by pos_system.py.
if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    gesture = HandGestureService()

    session_open = False
    phase = "WAIT_OPEN"

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        label = gesture.classify(frame)

        if label == "open" and phase == "WAIT_OPEN":
            phase = "WAIT_CLOSED"
        elif label == "closed" and phase == "WAIT_CLOSED":
            session_open = not session_open
            phase = "WAIT_OPEN"

        session_text = "SESSION: OPEN" if session_open else "SESSION: CLOSED"
        cv2.putText(
            frame,
            session_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )

        cv2.imshow("Session Toggle Gesture Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
