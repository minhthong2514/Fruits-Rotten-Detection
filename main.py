from ultralytics import YOLO
import cv2
import numpy as np
import time

def process_watershed(frame, mask):
    # Convert to grayscale and process only the object region
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Apply slight blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive threshold instead of Otsu
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 2
    )

    # Morphological preprocessing
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    # Distance transform to find sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.6 * dist_transform.max(), 255, 0
    )

    kernel_n = np.ones((3, 3), np.uint8)
    sure_fg = cv2.dilate(sure_fg, kernel_n, iterations=2)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_CLOSE, kernel_n, iterations=3)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed algorithm
    frame_copy = frame.copy()
    markers = cv2.watershed(frame_copy, markers)
    frame_copy[markers == -1] = [0, 0, 255]  # Mark boundaries in red

    return frame_copy, unknown


# Load YOLOv8 segmentation model
model = YOLO(
    r"F:\University\Nam_ba\Do_an_TGMT\YOLOv8-seg\yolov8n-seg-seg\runs\segment\train\weights\best.pt"
)

# Colors for each class ID
class_colors = {
    0: (0, 0, 255),      # Red (freshapple)
    1: (0, 255, 255),    # Yellow (freshbanana)
    2: (0, 165, 255),    # Orange (freshorange)
    3: (0, 0, 139),      # Dark red (rottenapple)
    4: (0, 140, 140),    # Dark yellow (rottenbanana)
    5: (0, 100, 200),    # Dark orange (rottenorange)
}

# Initialize time variable for FPS calculation
prev_frame_time = time.time()

# Use stream=True to read directly from webcam
for result in model.predict(source=1, stream=True, imgsz=640, conf=0.8, verbose=False):
    start_time = time.time()

    frame = result.orig_img
    frame_original = frame.copy()

    masks = result.masks
    boxes = result.boxes
    names = result.names

    # Binary mask image (single channel)
    mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
    cropped = np.zeros_like(frame, dtype=np.uint8)

    if masks is not None and boxes is not None:
        for i in range(len(masks.data)):
            mask = masks.data[i].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)

            # Resize mask to match frame size
            mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            cls_id = int(boxes[i].cls.item())
            color = class_colors.get(cls_id, (255, 255, 255))

            # Create colored mask for overlay
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask > 0, color[c], 0)

            # Overlay mask on the frame
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
            label = names[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

            # Apply watershed for rotten fruit
            watershed_img, thresh = process_watershed(frame_original, mask)
            cv2.imshow("Thresh", thresh)

            # mask_img = cv2.bitwise_or(mask_img, mask) 
            # cropped = cv2.bitwise_and(frame, frame, mask=mask_img) 
            # # print(cropped.shape)
            
            # If the detected object is rotten, overlay watershed result
            if "rotten" in label:
                alpha = 0.5  # Transparency for overlay
                frame = cv2.addWeighted(frame, 1 - alpha, watershed_img, alpha, 0)

    # Calculate and display FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(
        frame, fps_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

    # Display output
    cv2.imshow("YOLOv8 Segmentation Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
