import os
import numpy as np
from tqdm import tqdm
import cv2


def extract_dense_sift_descriptor(image_path, mask_path, step_size=10):
    image_rgb = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image_rgb is None or mask is None:
        return None

    # Convert mask to binary
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply mask to extract ROI
    roi = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_binary)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Generate dense keypoints within the masked region
    keypoints = [
        cv2.KeyPoint(x, y, step_size)
        for y in range(0, gray_roi.shape[0], step_size)
        for x in range(0, gray_roi.shape[1], step_size)
        if mask[y, x] > 0
    ]

    if not keypoints:
        return None

    # Compute SIFT descriptors
    _, descriptors = sift.compute(gray_roi, keypoints)
    if descriptors is None:
        return None

    # Return the mean descriptor (128-dimensional feature vector)
    return descriptors.mean(axis=0)


# Dataset directories
base_dir = r"F:\University\Nam_ba\Do_an_TGMT\SVM\dataset"
classes = ['fresh', 'rotten']
X, y = [], []

for label, cls in enumerate(classes):  # fresh: 0, rotten: 1
    raw_dir = os.path.join(base_dir, cls, "raw")
    mask_dir = os.path.join(base_dir, cls, "mask")

    filenames = os.listdir(raw_dir)
    for file in tqdm(filenames, desc=f"Processing {cls}"):
        raw_path = os.path.join(raw_dir, file)
        mask_path = os.path.join(mask_dir, file)

        feature = extract_dense_sift_descriptor(raw_path, mask_path)
        if feature is not None:
            X.append(feature)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total number of samples: {len(X)}")
