import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

from . import state  # talk to POS state

# =========================
# Configuration
# =========================

# Camera index: 0 is usually the default webcam.
CAMERA_INDEX = 0

# Target frame size (width, height) for processing.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Path to your YOLO model.
# Assumes best.pt is in the project root (same level as app/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "best.pt")  # adjust if needed

# Confidence threshold for detections
CONF_THRESHOLD = 0.9

# =========================
# Globals
# =========================

_model: Optional[YOLO] = None
_capture_thread: Optional[threading.Thread] = None
_stop_flag = False

_latest_frame_lock = threading.Lock()
_latest_frame: Optional[np.ndarray] = None

# MediaPipe Hands
mp_hands = mp.solutions.hands
_hands = None

# Gesture FSM state
gesture_state = "idle"
last_open_time = 0.0
last_toggle_time = 0.0

# Tunable parameters (for open → closed gesture)
GESTURE_SEQUENCE_TIMEOUT = 2.0        # max time between open and closed (sec)
GESTURE_TOGGLE_COOLDOWN = 2.0         # cooldown between toggles (sec)

# Require more stable gestures now
MIN_OPEN_FRAMES = 7                   # frames to confirm open palm
MIN_CLOSED_FRAMES = 7                 # frames to confirm closed fist

open_frame_count = 0
closed_frame_count = 0


# =========================
# Model + camera initialization
# =========================

def load_model():
    """Load YOLO and MediaPipe Hands if not already loaded."""
    global _model, _hands

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO model not found at {MODEL_PATH}. "
                f"Place your best.pt there or update MODEL_PATH in vision.py."
            )
        _model = YOLO(MODEL_PATH)
        print(f"[VISION] Loaded YOLO model from {MODEL_PATH}")

    if _hands is None:
        _hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,          # lightweight
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[VISION] MediaPipe Hands initialized.")


def start_camera():
    """Initialize the YOLO model and start the camera capture thread."""
    global _capture_thread, _stop_flag

    if _capture_thread is not None and _capture_thread.is_alive():
        print("[VISION] Camera thread already running.")
        return

    load_model()

    _stop_flag = False
    _capture_thread = threading.Thread(target=_camera_loop, daemon=True)
    _capture_thread.start()
    print("[VISION] Camera capture thread started.")


def stop_camera():
    """Signal the camera thread to stop."""
    global _stop_flag
    _stop_flag = True
    print("[VISION] Camera capture stop requested.")


# =========================
# Camera capture + YOLO loop
# =========================

def _camera_loop():
    """Continuously grab frames from camera, run YOLO + gestures, store last frame."""
    global _latest_frame

    print("[VISION] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[VISION] Failed to open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("[VISION] Camera opened successfully.")

    try:
        while not _stop_flag:
            ret, frame = cap.read()
            if not ret:
                print("[VISION] Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            annotated = _run_yolo_and_annotate(frame)

            with _latest_frame_lock:
                _latest_frame = annotated

            time.sleep(0.01)
    finally:
        cap.release()
        print("[VISION] Camera released.")


def _run_yolo_and_annotate(frame: np.ndarray) -> np.ndarray:
    """
    Run YOLO on the frame and draw bounding boxes.
    Also sends the BEST detection (if any) to state.process_detection().
    Runs hand-gesture logic every frame.
    """
    global _model
    annotated = frame.copy()

    if _model is None:
        state.process_detection(None, None)
        _process_hand_gesture(annotated)
        return annotated

    results = _model(frame, verbose=False)

    best_box = None
    best_conf = -1.0

    if results and len(results) > 0:
        res = results[0]

        if res.boxes is not None and len(res.boxes) > 0:
            # Pick highest-confidence detection for POS logic
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

            if best_box is not None:
                cls_id = int(best_box.cls[0])
                class_name = _model.names.get(cls_id, str(cls_id))
                state.process_detection(cls_id, class_name)
            else:
                state.process_detection(None, None)

            # Draw all boxes for visualization
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = _model.names.get(cls_id, str(cls_id))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - th - baseline),
                    (x1 + tw, y1),
                    (0, 255, 0),
                    thickness=-1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
        else:
            state.process_detection(None, None)
    else:
        state.process_detection(None, None)

    # Hand gesture overlay & session toggle
    _process_hand_gesture(annotated)

    return annotated


# =========================
# Hand gesture helpers
# =========================

def _classify_hand_gesture(hand_landmarks, image_width: int, image_height: int) -> str:
    """
    Stricter classification of hand gesture as 'open', 'closed', or 'unknown'.

    - 'open'   → all 4 fingers clearly extended
    - 'closed' → 0 fingers clearly extended
    - else     → 'unknown'
    """
    finger_tips = [8, 12, 16, 20]   # index, middle, ring, little tips
    finger_pips = [6, 10, 14, 18]   # PIP joints

    extended_fingers = 0

    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]

        # y is normalized [0,1]; smaller y = higher on image
        # stricter margin so half-bent fingers don't count
        if tip.y < pip.y - 0.05:
            extended_fingers += 1

    if extended_fingers >= 4:
        return "open"
    elif extended_fingers == 0:
        return "closed"
    else:
        return "unknown"


def _process_hand_gesture(frame: np.ndarray) -> str:
    """
    Run MediaPipe Hands, detect open → closed sequence, and toggle POS session
    using state.start_session() / state.end_session().
    """
    global _hands, gesture_state, last_open_time, last_toggle_time
    global open_frame_count, closed_frame_count

    if _hands is None:
        return "none"

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(frame_rgb)

    gesture_label = "none"
    now = time.time()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        gesture_label = _classify_hand_gesture(hand_landmarks, w, h)

        # (Optional) draw landmarks
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

        # FSM for open → closed sequence
        if gesture_label == "open":
            open_frame_count += 1
            closed_frame_count = 0

            if gesture_state == "idle" and open_frame_count >= MIN_OPEN_FRAMES:
                gesture_state = "open_confirmed"
                last_open_time = now

        elif gesture_label == "closed":
            closed_frame_count += 1

            if (
                gesture_state == "open_confirmed"
                and closed_frame_count >= MIN_CLOSED_FRAMES
                and (now - last_open_time) <= GESTURE_SEQUENCE_TIMEOUT
                and (now - last_toggle_time) >= GESTURE_TOGGLE_COOLDOWN
            ):
                # Toggle session via global state
                if state.pos_session.active:
                    state.end_session()
                else:
                    state.start_session()

                last_toggle_time = now
                gesture_state = "idle"
                open_frame_count = 0
                closed_frame_count = 0

        else:
            # unknown / noise: slowly decay counts
            open_frame_count = max(open_frame_count - 1, 0)
            closed_frame_count = max(closed_frame_count - 1, 0)

        # timeout for open_confirmed
        if (
            gesture_state == "open_confirmed"
            and (now - last_open_time) > GESTURE_SEQUENCE_TIMEOUT
        ):
            gesture_state = "idle"
            open_frame_count = 0
            closed_frame_count = 0

        cv2.putText(
            frame,
            f"Hand Gesture: {gesture_label}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    else:
        open_frame_count = max(open_frame_count - 1, 0)
        closed_frame_count = max(closed_frame_count - 1, 0)

        if (
            gesture_state == "open_confirmed"
            and (now - last_open_time) > GESTURE_SEQUENCE_TIMEOUT
        ):
            gesture_state = "idle"
            open_frame_count = 0
            closed_frame_count = 0

        gesture_label = "none"

    status_text = "SESSION ACTIVE" if state.pos_session.active else "SESSION INACTIVE"
    cv2.putText(
        frame,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return gesture_label


# =========================
# Helper for FastAPI video endpoint
# =========================

def get_latest_frame_jpeg() -> Optional[bytes]:
    """
    Returns the latest annotated frame encoded as JPEG bytes,
    or None if no frame is available yet.
    """
    with _latest_frame_lock:
        if _latest_frame is None:
            return None
        frame = _latest_frame.copy()

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None

    return buffer.tobytes()