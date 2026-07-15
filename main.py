import cv2
import torch
from ultralytics import YOLO


# Configuration
# ----------------------------

MODEL_NAME = "yolov8m.pt"

IMG_SIZE = 640
CONFIDENCE = 0.23

ZOOM_SIZE = 500
ZOOM_PADDING = 5
CROP_SIZE = 350

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 120

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


# Fullscreen

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


tracked_id = None


# Main Loop

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
        classes=[0,2,3],
        verbose=False
    )

    if len(results):

        boxes = results[0].boxes

        if boxes is not None and len(boxes):

            ids = boxes.id

            if ids is not None:

                ids = ids.int().cpu().tolist()

                current_target = None

                if tracked_id is not None:

                    for i, box in enumerate(boxes):

                        if ids[i] == tracked_id:
                            current_target = box
                            break

                if current_target is None:

                    largest_area = 0

                    for i, box in enumerate(boxes):

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        area = (x2-x1)*(y2-y1)

                        if area > largest_area:
                            largest_area = area
                            current_target = box
                            tracked_id = ids[i]


                if current_target is not None:

                    x1, y1, x2, y2 = map(int, current_target.xyxy[0])

                    cv2.rectangle(
                        annotated,
                        (x1,y1),
                        (x2,y2),
                        (255,0,0),
                        3
                    )

                    cv2.putText(
                        annotated,
                        f"",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,0,0),
                        2
                    )

                    cx = (x1+x2)//2
                    cy = (y1+y2)//2

                    left = max(0,cx-CROP_SIZE//2)
                    top = max(0,cy-CROP_SIZE//2)
                    right = min(frame.shape[1],cx+CROP_SIZE//2)
                    bottom = min(frame.shape[0],cy+CROP_SIZE//2)

                    crop = frame[top:bottom,left:right]

                    if crop.size:

                        zoom = cv2.resize(
                            crop,
                            (ZOOM_SIZE,ZOOM_SIZE),
                            interpolation=cv2.INTER_CUBIC
                        )

                        h,w = annotated.shape[:2]

                        overlay_x = w-ZOOM_SIZE-ZOOM_PADDING
                        overlay_y = h-ZOOM_SIZE-ZOOM_PADDING

                        cv2.rectangle(
                            annotated,
                            (overlay_x-2,overlay_y-2),
                            (overlay_x+ZOOM_SIZE+2,overlay_y+ZOOM_SIZE+2),
                            (255,255,255),
                            2
                        )

                        annotated[
                            overlay_y:overlay_y+ZOOM_SIZE,
                            overlay_x:overlay_x+ZOOM_SIZE
                        ] = zoom

    cv2.imshow("Detection", annotated)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()