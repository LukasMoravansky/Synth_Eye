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
synthetic = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\train\Image_001.png")
real = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\test\Image_001.png")

synthetic_rgb = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB)
real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

def yolo_to_pixel_box(yolo_box, img_shape):
    h, w = img_shape[:2]
    cx, cy, bw, bh = yolo_box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2

# ---- Transform steps ----

def apply_geometric_distortion(image, angle_range=2.0, scale_range=0.02, shift_range=5):
    h, w = image.shape[:2]
    angle = np.random.uniform(-angle_range, angle_range)
    scale = 1.0 + np.random.uniform(-scale_range, scale_range)
    tx = np.random.uniform(-shift_range, shift_range)
    ty = np.random.uniform(-shift_range, shift_range)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += [tx, ty]
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT101)

def add_realistic_noise(image, std=8):
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_color_style(image, alpha=0.95, beta=10):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    warm_factor = np.random.uniform(1.02, 1.08)
    cool_factor = np.random.uniform(0.92, 0.98)
    warm_filter = np.array([[[warm_factor, 1.0, cool_factor]]])
    warmed = np.clip(adjusted * warm_filter, 0, 255).astype(np.uint8)
    return warmed

def add_vignette(image, strength=0.3):
    rows, cols = image.shape[:2]

    # Generate global vignette mask
    kernel_x = cv2.getGaussianKernel(cols, cols * strength)
    kernel_y = cv2.getGaussianKernel(rows, rows * strength)
    kernel = kernel_y @ kernel_x.T  # outer product
    mask = kernel / np.max(kernel)

    # Apply to each channel
    vignette = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        vignette[:, :, i] = image[:, :, i] * mask

    return np.clip(vignette, 0, 255).astype(np.uint8)

def apply_blur(image, kernel=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(image, kernel, sigmaX=sigma)

def match_histogram_style(source, reference):
    return match_histograms(source, reference, channel_axis=-1).astype(np.uint8)

# ---- Full pipeline ----

def process_synthetic_image(image, real_ref):
    distorted = apply_geometric_distortion(image)
    noisy = add_realistic_noise(distorted)
    blurred_once = apply_blur(noisy, kernel=(3, 3), sigma=0.7)
    styled = adjust_color_style(blurred_once)

    vignetted = add_vignette(styled, strength=0.4)

    matched = match_histogram_style(vignetted, real_ref)
    final = apply_blur(matched, kernel=(5, 5), sigma=1.0)
    return final

# ---- Run the pipeline ----
final_result = process_synthetic_image(synthetic_rgb, real_rgb)

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
