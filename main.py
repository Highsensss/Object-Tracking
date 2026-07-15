import cv2
import torch
from ultralytics import YOLO


# Configuration
# ----------------------------

MODEL_NAME = "yolov8s.pt"

IMG_SIZE = 320
CONFIDENCE = 0.30

ZOOM_SIZE = 400
ZOOM_PADDING = 5
CROP_SIZE = 300
HALF_CROP = CROP_SIZE // 2

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 60

TRACK_CLASSES = [0, 2, 3]  

# GPU

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = YOLO(MODEL_NAME)
model.to(device)


# Camera

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")


FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
OVERLAY_X = FRAME_W - ZOOM_SIZE - ZOOM_PADDING
OVERLAY_Y = FRAME_H - ZOOM_SIZE - ZOOM_PADDING


# Fullscreen

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


tracked_id = None


def select_target(boxes, ids, tracked_id):
    xyxy = boxes.xyxy.cpu().numpy()

    if tracked_id is not None:
        try:
            idx = ids.index(tracked_id)
            return boxes[idx], xyxy[idx], tracked_id
        except ValueError:
            pass

    
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(areas.argmax())
    return boxes[idx], xyxy[idx], ids[idx]


# Main Loop

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        annotated = frame.copy()

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            classes=TRACK_CLASSES,
            verbose=False
        )

        if results:

            boxes = results[0].boxes

            if boxes is not None and len(boxes):

                ids = boxes.id

                if ids is not None:

                    ids = ids.int().cpu().tolist()

                    _, coords, tracked_id = select_target(boxes, ids, tracked_id)

                    x1, y1, x2, y2 = coords.astype(int)

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        3
                    )

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    left = max(0, cx - HALF_CROP)
                    top = max(0, cy - HALF_CROP)
                    right = min(frame.shape[1], cx + HALF_CROP)
                    bottom = min(frame.shape[0], cy + HALF_CROP)

                    crop = frame[top:bottom, left:right]

                    if crop.size:

                        zoom = cv2.resize(
                            crop,
                            (ZOOM_SIZE, ZOOM_SIZE),
                            interpolation=cv2.INTER_CUBIC
                        )

                        cv2.rectangle(
                            annotated,
                            (OVERLAY_X - 2, OVERLAY_Y - 2),
                            (OVERLAY_X + ZOOM_SIZE + 2, OVERLAY_Y + ZOOM_SIZE + 2),
                            (255, 0, 0),
                            2
                        )

                        annotated[
                            OVERLAY_Y:OVERLAY_Y + ZOOM_SIZE,
                            OVERLAY_X:OVERLAY_X + ZOOM_SIZE
                        ] = zoom

        cv2.imshow("Detection", annotated)

        key = cv2.waitKey(1)

        if key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()