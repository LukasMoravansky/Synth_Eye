import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# ---- Load images ----
synthetic = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\train\Image_001.png")
real = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\test\Image_001.png")

synthetic_rgb = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB)
real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

def apply_geometric_distortion(image, angle_range=2.0, scale_range=0.02, shift_range=5):
    h, w = image.shape[:2]

    # Random rotation
    angle = np.random.uniform(-angle_range, angle_range)

    # Random scale
    scale = 1.0 + np.random.uniform(-scale_range, scale_range)

    # Random translation
    tx = np.random.uniform(-shift_range, shift_range)
    ty = np.random.uniform(-shift_range, shift_range)

    # Build transformation matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += [tx, ty]  # apply translation

    distorted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return distorted

# ---- 1. Add camera-like sensor noise ----
def add_realistic_noise(image, mean=0, std=8):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# ---- 2. Add vignette effect (optional) ----
def add_vignette(image, strength=0.3):
    rows, cols = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols*strength)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows*strength)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette = np.empty_like(image)
    for i in range(3):
        vignette[:,:,i] = image[:,:,i] * mask
    return np.clip(vignette, 0, 255).astype(np.uint8)

# ---- 3. Reduce contrast slightly, warm up image ----
def adjust_color_style(image, alpha=0.95, beta=10):
    # Slightly reduce contrast and warm up tones
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    warm_factor = np.random.uniform(1.02, 1.08)
    cool_factor = np.random.uniform(0.92, 0.98)
    warm_filter = np.array([[[warm_factor, 1.0, cool_factor]]])
    warmed = np.clip(adjusted * warm_filter, 0, 255).astype(np.uint8)
    return warmed

# ---- 4. Match histogram (color profile) to real image ----
def match_histogram_style(source, reference):
    matched = match_histograms(source, reference, channel_axis=-1)
    return matched.astype(np.uint8)

# ---- 5. Slight Gaussian blur to simulate optics ----
def apply_blur(image):
    return cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)

# ---- Apply all steps ----
step0 = apply_geometric_distortion(synthetic_rgb)
step1 = add_realistic_noise(step0, std=8)
step2 = adjust_color_style(step1)
#step3 = add_vignette(step2, strength=0.4)
step4 = match_histogram_style(step2, real_rgb)
final = apply_blur(step4)

# ---- Display result ----
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