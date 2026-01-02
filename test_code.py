from ultralytics import YOLO
import cv2
import numpy as np
import time

prev_sure_fg = None
def process_watershed(frame, mask):
    global prev_sure_fg
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)

    kernel_n = np.ones((5, 5), np.uint8)
    sure_fg = cv2.dilate(sure_fg, kernel_n, iterations=2)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, kernel_n, iterations=4)

    if prev_sure_fg is not None:
        sure_fg = cv2.addWeighted(sure_fg.astype(np.uint8), 0.7, prev_sure_fg.astype(np.uint8), 0.3, 0)
    prev_sure_fg = sure_fg.copy()

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    frame_copy = frame.copy()
    markers = cv2.watershed(frame_copy, markers)
    frame_copy[markers == -1] = [0, 0, 255]

    return frame_copy, sure_fg

# Load model YOLOv8 segmentation
model = YOLO(r"F:\University\Nam_ba\Do_an_TGMT\YOLOv8-seg\yolov8n-seg\runs\segment\train\weights\best.pt")

# Màu theo class ID
class_colors = {
    0: (0, 0, 255),
    1: (0, 255, 255),
    2: (0, 165, 255),
    3: (0, 0, 139),
    4: (0, 140, 140),
    5: (0, 100, 200),
}

prev_frame_time = time.time()

# Stream webcam
for result in model.predict(source=1, stream=True, imgsz=640, conf=0.8, verbose=False):
    frame = result.orig_img
    frame_original = frame.copy()
    masks = result.masks
    boxes = result.boxes
    names = result.names

    if masks is not None and boxes is not None:
        for i in range(len(masks.data)):
            mask = masks.data[i].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            cls_id = int(boxes[i].cls.item())
            color = class_colors.get(cls_id, (255, 255, 255))
            label = names[cls_id]

            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask > 0, color[c], 0)

            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
            x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if "rotten" in label:
                watershed_img, sure_fg = process_watershed(frame_original, mask)
                alpha = 0.5
                frame = cv2.addWeighted(frame, 1 - alpha, watershed_img, alpha, 0)
                cv2.imshow("Sure Foreground", sure_fg)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Segmentation Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
