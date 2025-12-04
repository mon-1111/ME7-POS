import cv2
import mediapipe as mp

from camera_service import CameraService


class HandGestureService:
    """
    Uses MediaPipe Hands to classify a hand as 'open' or 'closed'
    based on how many fingers are extended.
    """

    def __init__(self):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.enabled = True

    def set_enabled(self, flag: bool):
        self.enabled = flag

    def classify(self, frame):
        """
        Return 'open', 'closed', or None using MediaPipe Hands.
        """
        if not self.enabled:
            return None

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            return None

        hand = result.multi_hand_landmarks[0].landmark

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


if __name__ == "__main__":
    cam = CameraService()
    gesture = HandGestureService()

    # Session toggle state
    session_open = False
    phase = "WAIT_OPEN"   # WAIT_OPEN → WAIT_CLOSED → toggle

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        label = gesture.classify(frame)

        # ---- open → closed toggles session ----
        if label == "open" and phase == "WAIT_OPEN":
            phase = "WAIT_CLOSED"

        elif label == "closed" and phase == "WAIT_CLOSED":
            session_open = not session_open
            phase = "WAIT_OPEN"
        # ---------------------------------------

        # Draw only the session state
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

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()