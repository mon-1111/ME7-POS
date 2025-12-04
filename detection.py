import time
import cv2
from ultralytics import YOLO

from camera_service import CameraService
from csv_manager import ItemCatalog


MODEL_PATH = r"/Users/rjbagunu/Desktop/Grad School (PhD AI) /AI 231/Machine Exercises/ME7 POS/best.pt"
CONF_THRESHOLD = 0.5  # you can adjust later


def main():
    # Load model
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # Load item catalog
    catalog = ItemCatalog()

    # Start camera
    cam = CameraService()

    prev_time = time.time()

    while True:
        frame = cam.read_frame()
        if frame is None:
            print("Failed to read frame.")
            break

        # Run YOLO inference
        results = model(frame, verbose=False)[0]  # first (and only) result

        # Draw detections
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            meta = catalog.get(cls_id)
            if meta is not None:
                label = f"{meta['product']} {conf:.2f}"
            else:
                label = f"ID {cls_id} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # FPS display (optional)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("YOLOv8 POS Item Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()