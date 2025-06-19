import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import sys
# Add access to the custom utilities
if '../../src' not in sys.path:
    sys.path.append('../../src')
import Utilities.File_IO as File_IO

# Image_Background_001
# ---- Load synthetic and real images ----
synthetic_path = r"C:\projects\Synth_Eye\Data\Dataset_v1\images\train\Image_001.png"
real_path = r"C:\projects\Synth_Eye\Data\Dataset_v1\images\test\Image_001.png"
label_path = r"C:\projects\Synth_Eye\Data\Dataset_v1\labels\train\Image_001"
label_data = File_IO.Load(label_path, 'txt', ' ')

synthetic = cv2.imread(synthetic_path)
real = cv2.imread(real_path)

# Convert to RGB for consistent color processing
synthetic_rgb = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB)
real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

def yolo_to_pixel_box(yolo_box, image_shape):
    """Convert normalized YOLO box to pixel coordinates."""
    _, x_c, y_c, w, h = yolo_box
    img_h, img_w = image_shape[:2]

    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)

    return x1, y1, x2, y2

def add_rust_blobs_from_yolo(image_rgb, yolo_box, num_blobs=5):
    h, w = image_rgb.shape[:2]
    rust_color = np.array([110, 55, 25], dtype=np.uint8)

    # Convert YOLO to pixel coordinates
    x1, y1, x2, y2 = yolo_to_pixel_box(yolo_box, image_rgb.shape)
    rust_mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(num_blobs):
        blob_w = np.random.randint(0.3*(x2-x1), 0.8*(x2-x1))
        blob_h = np.random.randint(0.2*(y2-y1), 0.6*(y2-y1))
        cx = np.random.randint(x1, x2)
        cy = np.random.randint(y1, y2)

        noise = np.random.normal(127, 40, (blob_h, blob_w)).astype(np.uint8)
        _, blob = cv2.threshold(noise, 128, 255, cv2.THRESH_BINARY)

        # Resize and blend into rust_mask
        mask_resized = np.zeros_like(rust_mask)
        x_start = max(cx - blob_w // 2, 0)
        y_start = max(cy - blob_h // 2, 0)
        x_end = min(x_start + blob_w, w)
        y_end = min(y_start + blob_h, h)

        blob_crop = blob[:y_end - y_start, :x_end - x_start]
        mask_resized[y_start:y_end, x_start:x_end] = blob_crop

        rust_mask = cv2.bitwise_or(rust_mask, mask_resized)

    rust_mask = cv2.GaussianBlur(rust_mask, (35, 35), sigmaX=20)
    rust_mask_norm = rust_mask.astype(np.float32) / 255.0
    rust_mask_norm = np.expand_dims(rust_mask_norm, axis=2)

    rust_layer = (rust_mask_norm * rust_color).astype(np.uint8)
    final = cv2.addWeighted(image_rgb, 1.0, rust_layer, 0.5, 0)

    return final

# ---- Step 2: Add realistic noise ----
def add_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# ---- Step 3: Adjust contrast/brightness ----
def adjust_contrast_brightness(image, alpha=1.2, beta=10):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# ---- Step 4: Histogram matching to real ----
def match_histogram_style(source, reference):
    matched = match_histograms(source, reference, channel_axis=-1)
    return matched.astype(np.uint8)

# ---- Step 5: Slight blur ----
def apply_blur(image):
    return cv2.GaussianBlur(image, (5, 5), sigmaX=1)

# ---- Apply transformations ----
yolo_box = label_data[0,:]
modified_img = add_rust_blobs_from_yolo(synthetic_rgb, yolo_box)
noisy = add_noise(modified_img)
adjusted = adjust_contrast_brightness(noisy)
matched = match_histogram_style(adjusted, real_rgb)
final = apply_blur(matched)

# ---- Display comparison ----
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Synthetic Image")
plt.imshow(synthetic_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Modified Synthetic Image")
plt.imshow(final)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Real-World Image")
plt.imshow(real_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()

# ---- Save result ----
#output_path = r"C:\projects\Synth_Eye\Data\Dataset_v1\images\train\Image_001_modified.png"
#cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
#print(f"✅ Modified synthetic image saved to: {output_path}")
