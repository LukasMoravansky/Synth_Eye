import cv2
import numpy as np
import sys
import os
import numpy as np
import Utils

SRC_PATH = os.path.abspath('../../src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from pathlib import Path
import re

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

def main():
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    for partition_name in ['test', 'train', 'valid']:
        folder_path = f'{project_folder}/Data/Dataset_v1/images/{partition_name}'
        file_info_list = extract_numbers_from_filenames(folder_path)
        for file_info in file_info_list:
            image_data_tmp = cv2.imread(file_info['path'])

            if image_data_tmp is None:
                raise FileNotFoundError(f"Unable to load image from: {file_info['path']}")

            if partition_name in ['train', 'valid']:
                image_data = process_synthetic_image(image_data_tmp)
            else:
                image_data = image_data_tmp.copy()

            cv2.imwrite(f'{project_folder}/Data/Dataset_v2/images/{partition_name}/{file_info["name"]}.png', image_data.copy())

if __name__ == '__main__':
    main()