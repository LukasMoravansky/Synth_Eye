from pathlib import Path
import re
import numpy as np
import cv2

def extract_numbers_from_filenames(folder_path):
    folder = Path(folder_path)
    results = []
    
    for file in folder.iterdir():
        if file.is_file() and file.suffix == '.png':
            match = re.search(r'Image_((?:\w+_)*)(\d+)\.png', file.name)
            if match:
                prefix = match.group(1)
                number = int(match.group(2))
                is_background = 'Background' in prefix
                results.append({
                    'name': file.stem,
                    'id': number,
                    'is_background': is_background,
                    'path': str(file)
                })
    
    return results

def yolo_to_absolute(x_c, y_c, w, h, img_w, img_h):
    abs_x = x_c * img_w
    abs_y = y_c * img_h
    abs_w = w * img_w
    abs_h = h * img_h
    return abs_x, abs_y, abs_w, abs_h

def absolute_to_yolo(x, y, w, h, img_w, img_h):
    return x / img_w, y / img_h, w / img_w, h / img_h

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


def process_real_image(image: np.ndarray) -> np.ndarray:
    return adjust_color_style(image)
