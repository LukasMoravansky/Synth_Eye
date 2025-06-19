import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import sys
# Add access to the custom utilities
if '../../src' not in sys.path:
    sys.path.append('../../src')
import Utilities.File_IO as File_IO

# ---- Load images ----
synthetic = cv2.imread(r"C:\projects\Synth_Eye\Evaluation\Dataset\Image_002.png")
real = cv2.imread(r"C:\projects\Data\Dataset_v1\images\test\Image_039.png")

synthetic_rgb = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB)
real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

def add_realistic_noise(image, std=5):
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_color_style(image, alpha=0.95, beta=5):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    warm_filter = np.array([[[1.0, 1.0, 0.98]]])
    warmed = np.clip(adjusted * warm_filter, 0, 255).astype(np.uint8)
    return warmed

def apply_blur(image, kernel=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(image, kernel, sigmaX=sigma)

def process_synthetic_image(image):
    noisy = add_realistic_noise(image)
    blurred_once = apply_blur(noisy, kernel=(3, 3), sigma=0.5)
    styled = adjust_color_style(blurred_once)
    final = apply_blur(styled, kernel=(3, 3), sigma=1.0)
    return final

# ---- Run the pipeline ----
final_result = process_synthetic_image(synthetic_rgb)

# ---- Show result ----
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Synthetic Image")
plt.imshow(synthetic_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Modified Synthetic Image")
plt.imshow(final_result)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Real-World Image")
plt.imshow(real_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()
