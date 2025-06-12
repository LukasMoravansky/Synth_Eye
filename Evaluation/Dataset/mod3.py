import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
# Add access to the custom utilities
if '../../src' not in sys.path:
    sys.path.append('../../src')

# ---- Load images ----
synthetic = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\train\Image_Background_001.png")
real = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\test\Image_Background_001.png")

synthetic_rgb = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB)
real_rgb = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

def estimate_alpha_beta(source_gray, target_gray):
    src_mean = source_gray.mean()
    src_std = source_gray.std()
    tgt_mean = target_gray.mean()
    tgt_std = target_gray.std()
    
    alpha = tgt_std / (src_std + 1e-6)  # avoid div by zero
    beta = tgt_mean - alpha * src_mean
    return alpha, beta

def estimate_hsv_adjustments(source_hsv, target_hsv):
    # Calculate mean for each channel
    src_h_mean = source_hsv[..., 0].mean()
    src_s_mean = source_hsv[..., 1].mean()
    src_v_mean = source_hsv[..., 2].mean()
    
    tgt_h_mean = target_hsv[..., 0].mean()
    tgt_s_mean = target_hsv[..., 1].mean()
    tgt_v_mean = target_hsv[..., 2].mean()
    
    # Hue shift (mod 180)
    hue_shift = (tgt_h_mean - src_h_mean) % 180
    
    # Saturation multiplier
    saturation_mult = tgt_s_mean / (src_s_mean + 1e-6)
    
    # Value multiplier
    value_mult = tgt_v_mean / (src_v_mean + 1e-6)
    
    return {'hue': hue_shift, 'saturation': saturation_mult, 'value': value_mult}

def estimate_all_params(source_img, target_img):
    # Convert to grayscale for alpha, beta
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    alpha, beta = estimate_alpha_beta(source_gray, target_gray)
    
    # Convert to HSV for hue, saturation, value
    source_hsv = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_adjust = estimate_hsv_adjustments(source_hsv, target_hsv)
    
    return alpha, beta, hsv_adjust

# --- Add noise ---
def add_realistic_noise(image, std=8):
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# --- Apply Gaussian blur ---
def apply_blur(image, kernel=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(image, kernel, sigmaX=sigma)

# --- Adjust color using alpha, beta, and HSV shifts ---
def adjust_color_style(image, alpha=1.0, beta=0, warm_filter=True, hsv_adjust=None):
    # 1. Brightness/contrast
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 2. Warm filter
    if warm_filter:
        warm_filter_array = np.array([[[1.0, 1.0, 0.95]]])  # Reduce blue channel
        adjusted = np.clip(adjusted.astype(np.float32) * warm_filter_array, 0, 255).astype(np.uint8)
    
    # 3. HSV adjustment
    if hsv_adjust:
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_shift = hsv_adjust.get('hue', 0)
        s_mult = hsv_adjust.get('saturation', 1.0)
        v_mult = hsv_adjust.get('value', 1.0)
        
        hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * s_mult, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * v_mult, 0, 255)
        
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return adjusted

# --- Full synthetic image pipeline ---
def process_synthetic_image(image, alpha=1.0, beta=0, hsv_adjust=None):
    noisy = add_realistic_noise(image)
    blurred_once = apply_blur(noisy, kernel=(3, 3), sigma=0.7)
    styled = adjust_color_style(blurred_once, alpha=alpha, beta=beta, hsv_adjust=hsv_adjust)
    final = apply_blur(styled, kernel=(5, 5), sigma=1.0)
    return final

alpha, beta, hsv_adjust = estimate_all_params(synthetic_rgb, real_rgb)
final_result = process_synthetic_image(synthetic_rgb, alpha, beta, hsv_adjust)

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
