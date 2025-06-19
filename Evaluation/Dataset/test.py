import cv2
import numpy as np
import sys
import os

# Ensure custom utility path is included
SRC_PATH = os.path.abspath('../../src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ---- Load Image ----
INPUT_IMAGE_PATH = r"C:\projects\Data\Dataset_v1\images\train\Image_001.png"
OUTPUT_IMAGE_PATH = r"Image_001_processed.png"

image = cv2.imread(INPUT_IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Unable to load image from: {INPUT_IMAGE_PATH}")

# ---- Image Processing Utilities ----
def add_realistic_noise(image: np.ndarray, std_dev: float = 5.0) -> np.ndarray:
    """Adds Gaussian noise to an image."""
    noise = np.random.normal(0, std_dev, image.shape).astype(np.int16)
    noisy_image = image.astype(np.int16) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def adjust_color_style(image: np.ndarray, alpha: float = 0.95, beta: int = 5) -> np.ndarray:
    """Adjusts brightness/contrast and applies a warm filter."""
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    warm_filter = np.array([[[1.0, 1.0, 1.0]]])
    warmed = np.clip(adjusted * warm_filter, 0, 255).astype(np.uint8)
    return warmed

def apply_blur(image: np.ndarray, kernel_size: tuple = (5, 5), sigma: float = 1.0) -> np.ndarray:
    """Applies Gaussian blur to the image."""
    return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma)

def process_synthetic_image(image: np.ndarray) -> np.ndarray:
    """Applies a pipeline of noise addition, color adjustment, and blurring."""
    noisy = add_realistic_noise(image)
    blurred_once = apply_blur(noisy, kernel_size=(3, 3), sigma=0.5)
    styled = adjust_color_style(blurred_once)
    final_output = apply_blur(styled, kernel_size=(3, 3), sigma=1.0)
    return final_output

# ---- Process and Save Result ----
processed_image = process_synthetic_image(image)
cv2.imwrite(OUTPUT_IMAGE_PATH, processed_image)
print(f"Processed image saved to: {OUTPUT_IMAGE_PATH}")


