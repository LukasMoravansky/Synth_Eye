import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import match_histograms

def estimate_image_properties(image):
    # Convert to grayscale for brightness and contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)

    # Convert to HSV for saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])

    return brightness, contrast, saturation

def apply_image_properties(image, brightness_delta=0.0, contrast_ratio=1.0, saturation_ratio=1.0):
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Adjust L channel
    l = lab[:, :, 0]
    l = contrast_ratio * l + brightness_delta
    l = np.clip(l, 0, 255)
    lab[:, :, 0] = l

    img_bc = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Adjust saturation in HSV
    hsv = cv2.cvtColor(img_bc, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation_ratio
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result

def match_image_style(source_img, target_img):
    src_b, src_c, src_s = estimate_image_properties(source_img)
    tgt_b, tgt_c, tgt_s = estimate_image_properties(target_img)
    
    print(f"Source - Brightness: {src_b:.2f}, Contrast: {src_c:.2f}, Saturation: {src_s:.2f}")
    print(f"Target - Brightness: {tgt_b:.2f}, Contrast: {tgt_c:.2f}, Saturation: {tgt_s:.2f}")

    epsilon = 1e-5  # small number to avoid div by zero

    # Brightness delta: straightforward difference
    brightness_delta = src_b - tgt_b

    # Contrast ratio: avoid div zero & clip to sane range (e.g. 0.5x to 2x)
    contrast_ratio = src_c / (tgt_c + epsilon)
    contrast_ratio = np.clip(contrast_ratio, 0.5, 2.0)

    # Saturation ratio: similarly handle zero saturation target, clip too
    saturation_ratio = src_s / (tgt_s + epsilon)
    saturation_ratio = np.clip(saturation_ratio, 0.5, 2.0)

    # Optionally, if source values are too low (near zero), set ratio to 1 to avoid weird scaling
    if src_c < epsilon:
        contrast_ratio = 1.0
    if src_s < epsilon:
        saturation_ratio = 1.0

    print(brightness_delta, contrast_ratio, saturation_ratio)

    return apply_image_properties(target_img, -5.0, 1.0, 1.0)

# --- Load source and target images ---
real_img = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\test\Image_001.png")
synthetic_img = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\train\Image_001.png")

modified_img = match_image_style(real_img, synthetic_img)

# ---- Show result ----
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Synthetic Image")
plt.imshow(cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Modified Synthetic Image")
plt.imshow(cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Real-World Image")
plt.imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
