# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OpenCV library for computer vision tasks
import cv2
# OS module for file handling and accessing directories
import os
# Custom Lib.:
#   ../Basler/Camera
from Basler.Camera import Basler_Cls
#   ../Parameters/Scene
import Parameters.Scene

import numpy as np

# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 51

# Camera calibration matrix.
CAMERA_CALIBRATION_MATRIX = np.array([[7163.39148, 0.0, 884.750383], [0.0, 7185.41499, 491.271296], [0.0, 0.0, 1.0]], dtype=np.float64)

# Distortion coefficients: [k1, k2, p1, p2, k3].
CAMERA_CALIBRATION_DIST_COEFFS = np.array([-0.378556984, 28.1374127, -0.00651131765, -0.00121823652, 0.560603312], dtype=np.float64)

"""
PIXEL_TO_MM_X = 0.09744  # horizontal
PIXEL_TO_MM_Y = 0.09730  # vertical

def pixels_to_mm(x_px, y_px):
    return x_px * PIXEL_TO_MM_X, y_px * PIXEL_TO_MM_Y
"""

def main():
    """
    Description:
        A program to configure a Basler camera (a2A1920-51gcPRO) with custom settings to capture an image. The system is equipped
        with the EFFI-FD-200-200-000 lighting for optimal illumination in the vision stand.

        Setup:
            Camera Model: Basler a2A1920-51gcPRO GigE Camera
            Lighting: EFFI-FD-200-200-000 High-Power Flat Light
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Custom camera configuration.
    custom_cfg = {
        'exposure_time': 10000,
        'gain': 10,
        'balance_ratios': {'Red': 0.95, 'Green': 0.9, 'Blue': 1.2},
        'pixel_format': 'BayerRG8'
    }

    # Initialize and configure the Basler camera.
    Basler_Cam_Id_1 = Basler_Cls(config=custom_cfg)

    # Capture a single image.
    img_raw = Basler_Cam_Id_1.Capture()
    if img_raw is None:
        raise ValueError('No image captured!')

    # Release the camera resources.
    del Basler_Cam_Id_1

    # Define output image path.
    output_path = os.path.join(f'{project_folder}/Data/Camera/{Parameters.Scene.Basler_Cam_Str.Name}/', f'Image_Real_{(CONST_INIT_INDEX):03}.png')

    # Undistort the image using calibration data
    h, w = img_raw.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(CAMERA_CALIBRATION_MATRIX, CAMERA_CALIBRATION_DIST_COEFFS, (w, h), 1, (w, h))
    img_undistorted = cv2.undistort(img_raw, CAMERA_CALIBRATION_MATRIX, CAMERA_CALIBRATION_DIST_COEFFS, None, new_camera_matrix)

    # Save the image with bounding boxes.
    cv2.imwrite(output_path, img_undistorted)

    print(f'Result saved at: {output_path}')

if __name__ == '__main__':
    sys.exit(main())