import cv2
from ultralytics import YOLO
import os

from audio_service import AudioService
from camera_service import CameraService
from csv_manager import ItemCatalog
from cart_manager import CartManager
from hand_gesture import HandGestureService


MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
CONF_THRESHOLD = 0.7


def print_receipt(cart: CartManager):
    print("\n===== RECEIPT =====")
    lines = cart.get_lines()
    if not lines:
        print("(no items)")
    else:
        for line in lines:
            print(f"{line['product']:30} x{line['qty']:2} = {line['subtotal']:.2f}")
    print(f"TOTAL: PHP {cart.get_total():.2f}")
    print("===================")


def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    catalog = ItemCatalog()
    cart = CartManager(catalog)
    audio = AudioService()
    cam = CameraService()
    gesture = HandGestureService()

    # ------------------------------
    # SESSION CONTROL (GESTURE)
    # ------------------------------
    session_open = False
    phase = "WAIT_OPEN"

    last_gesture_label = None
    same_count = 0
    stable_label = None

    # Slightly more sensitive gesture
    MIN_STABLE_FRAMES = 4
    toggle_cooldown = 0
    TOGGLE_COOLDOWN_FRAMES = 20

    # ------------------------------
    # ITEM DETECTION / TRACKING
    # ------------------------------
    item_present = False          # True if any item is visible this frame
    counted_tracks = set()        # set of (class_id, track_id) we've already added to cart

    # ------------------------------
    # UI / SUMMARY
    # ------------------------------
    PANEL_WIDTH = 320
    show_summary = False          # whether to draw big summary banner
    summary_total = 0.0           # last session's total

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        height, width, _ = frame.shape

        # ============================================================
        # 1) GESTURE — USE ONLY CAMERA REGION, NOT EXTENDED FRAME
        # ============================================================
        gesture_view = frame[:, :width]
        raw_label = gesture.classify(gesture_view)

        if raw_label is None:
            same_count = 0
            stable_label = None
            last_gesture_label = None
        else:
            if raw_label == last_gesture_label:
                same_count += 1
            else:
                same_count = 1
                last_gesture_label = raw_label

            stable_label = raw_label if same_count >= MIN_STABLE_FRAMES else None

        prev_session_open = session_open

        # ============================================================
        # 2) ITEM DETECTION WITH TRACKING (only when session is OPEN)
        # ============================================================
        item_present = False  # reset; will be set True if we see any box

        if session_open:
            # Use tracking so items keep a stable track ID across frames
            results = model.track(
                frame,
                persist=True,
                conf=CONF_THRESHOLD,
                verbose=False
            )[0]

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD:
                    continue

                cid = int(box.cls[0])  # class id from YOLO

                # Track ID from the tracker
                track_id = None
                if hasattr(box, "id") and box.id is not None:
                    track_id = int(box.id[0])

                # Mark that some item is visible this frame
                item_present = True

                # Draw bounding box + label (optionally show track id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                meta = catalog.get(cid)
                label = meta["product"] if meta else f"ID {cid}"

                display_label = label if track_id is None else f"{label} #{track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    display_label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # ------------------- COUNT ONCE PER TRACK -------------------
                if track_id is not None:
                    key = (cid, track_id)
                    if key not in counted_tracks:
                        counted_tracks.add(key)
                        cart.add_item(cid)
                        meta = catalog.get(cid)
                        name = meta["product"] if meta else f"ID {cid}"
                        print(f"Added: {name}")
                        audio.play_beep()
                else:
                    # Fallback: if no track id (should be rare), skip to avoid spam.
                    pass

        else:
            item_present = False

        # ============================================================
        # 3) SESSION TOGGLE
        #    - Only allow OPEN when no item is visible
        #    - Allow CLOSE even if items are still visible
        # ============================================================
        if toggle_cooldown > 0:
            toggle_cooldown -= 1
        else:
            if not session_open:
                # CLOSED → want to OPEN (require clear view, no items)
                if not item_present:
                    if stable_label == "open" and phase == "WAIT_OPEN":
                        phase = "WAIT_CLOSED"
                    elif stable_label == "closed" and phase == "WAIT_CLOSED":
                        session_open = True
                        phase = "WAIT_OPEN"
                        toggle_cooldown = TOGGLE_COOLDOWN_FRAMES
            else:
                # OPEN → want to CLOSE (allow even if items present)
                if stable_label == "open" and phase == "WAIT_OPEN":
                    phase = "WAIT_CLOSED"
                elif stable_label == "closed" and phase == "WAIT_CLOSED":
                    session_open = False
                    phase = "WAIT_OPEN"
                    toggle_cooldown = TOGGLE_COOLDOWN_FRAMES

        # ============================================================
        # 4) START / END SESSION ACTIONS
        # ============================================================
        if session_open and not prev_session_open:
            # New session: clear previous cart + hide old summary + reset tracks
            cart.clear()
            counted_tracks.clear()
            item_present = False
            show_summary = False
            audio.play_beep()
            print("\n=== SESSION STARTED ===")

        if not session_open and prev_session_open:
            # Session just ended: store summary + print to console
            audio.play_beep()
            summary_total = cart.get_total()
            show_summary = True
            print("\n=== SESSION ENDED ===")
            print_receipt(cart)

            # OPTIONAL TTS announcement of total (if AudioService supports it)
            if hasattr(audio, "speak"):
                try:
                    audio.speak(f"Your total is {summary_total:.2f} pesos.")
                except Exception as e:
                    print(f"[Warning] TTS failed: {e}")

        # ============================================================
        # 5) DRAW RECEIPT PANEL
        # ============================================================
        extended_frame = cv2.copyMakeBorder(
            frame, 0, 0, 0, PANEL_WIDTH,
            cv2.BORDER_CONSTANT, value=(40, 40, 40)
        )

        # Background
        cv2.rectangle(
            extended_frame,
            (width, 0),
            (width + PANEL_WIDTH, height),
            (40, 40, 40),
            -1
        )

        # Session
        cv2.putText(
            extended_frame,
            "SESSION: OPEN" if session_open else "SESSION: CLOSED",
            (width + 20, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # Title
        cv2.putText(
            extended_frame,
            "RECEIPT",
            (width + 20, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # List items
        y_offset = 130
        for line in cart.get_lines():
            text = f"{line['product'][:16]:16} x{line['qty']}"
            cv2.putText(
                extended_frame,
                text,
                (width + 20, y_offset),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            y_offset += 28

        # Running total
        cv2.putText(
            extended_frame,
            f"TOTAL: PHP {cart.get_total():.2f}",
            (width + 20, height - 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        # ============================================================
        # 6) END-OF-SESSION SUMMARY BANNER
        # ============================================================
        if show_summary:
            banner_text = f"TOTAL DUE: PHP {summary_total:.2f}"

            # Draw semi-opaque rectangle across bottom of camera area
            banner_y1 = height - 80
            banner_y2 = height - 20
            overlay = extended_frame.copy()
            cv2.rectangle(
                overlay,
                (20, banner_y1),
                (width - 20, banner_y2),
                (0, 0, 0),
                -1,
            )
            alpha = 0.6
            extended_frame = cv2.addWeighted(
                overlay, alpha, extended_frame, 1 - alpha, 0
            )

            # Banner text
            cv2.putText(
                extended_frame,
                banner_text,
                (40, height - 35),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

        # ============================================================
        # 7) SHOW UI
        # ============================================================
        cv2.imshow("POS System (Camera + Receipt Panel)", extended_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            print_receipt(cart)
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()