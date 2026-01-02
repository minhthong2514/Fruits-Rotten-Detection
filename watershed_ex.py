import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Read the image
img = cv2.imread(r"imgs/chuoihu.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(12, 8))

# Step 2: Binary thresholding using Otsu's method
_, thresh = cv2.threshold(
    img_gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
plt.subplot(231)
plt.title("Binary Threshold (Otsu)")
plt.imshow(thresh, cmap='gray')

# Step 3: Dilation to obtain sure background
kernel = np.ones((3, 3), np.uint8)
sure_bg = cv2.dilate(thresh, kernel, iterations=3)
plt.subplot(232)
plt.title("Sure Background")
plt.imshow(sure_bg, cmap='gray')

# Step 4: Distance transform to obtain foreground information
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
plt.subplot(233)
plt.title("Distance Transform")
plt.imshow(dist_transform, cmap='gray')

# Step 5: Threshold the distance transform to obtain sure foreground
_, sure_fg = cv2.threshold(
    dist_transform, 0.5 * dist_transform.max(), 255, 0
)
sure_fg = np.uint8(sure_fg)
plt.subplot(234)
plt.title("Sure Foreground")
plt.imshow(sure_fg, cmap='gray')

# Step 6: Unknown region = background - foreground
unknown = cv2.subtract(sure_bg, sure_fg)
plt.subplot(235)
plt.title("Unknown Region")
plt.imshow(unknown, cmap='gray')

# Step 7: Label connected components in the foreground
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  # Background is labeled as 1, foreground starts from 2
markers[unknown == 255] = 0  # Unknown regions are labeled as 0

# Step 8: Apply the Watershed algorithm
markers = cv2.watershed(img_rgb, markers)
print(markers)

# Draw watershed boundaries (-1) in red
img_result = img_rgb.copy()
img_result[markers == -1] = [255, 0, 0]
plt.subplot(236)
plt.title("Watershed Result")
plt.imshow(img_result)

plt.tight_layout()
plt.show()
