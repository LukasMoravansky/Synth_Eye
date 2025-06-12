import sys
import os
import cv2
import numpy as np

# Add access to the custom utilities
if '../../src' not in sys.path:
    sys.path.append('../../src')

CONST_INIT_INDEX = 9

# Camera calibration matrix.
CAMERA_CALIBRATION_MATRIX = np.array([[4.82884536e+05, 0.0, 9.59366867e+02],[0.0, 4.82591133e+05, 5.99509961e+02],[0.0, 0.0, 1.0]], dtype=np.float64)

# Distortion coefficients: [k1, k2, p1, p2, k3].
CAMERA_CALIBRATION_DIST_COEFFS = np.array([28.5090395, 5.70509061e-05, 8.33112699e-04, 1.45115877e-02, 9.97217145e-11], dtype=np.float64)

def main():
    # Locate the project folder
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Load the image and label
    image_path = f'{project_folder}/Data/Dataset_v1/images/train/Image_{CONST_INIT_INDEX:03}.png'

    image_data = cv2.imread(image_path)

    # Undistort the image using calibration data
    h, w = image_data.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(CAMERA_CALIBRATION_MATRIX, CAMERA_CALIBRATION_DIST_COEFFS, (w, h), 1, (w, h))
    img_undistorted = cv2.undistort(image_data, CAMERA_CALIBRATION_MATRIX, CAMERA_CALIBRATION_DIST_COEFFS, None, new_camera_matrix)

    cv2.imwrite(f'Image_Res_{CONST_INIT_INDEX:03}_Original.png', img_undistorted)
if __name__ == '__main__':
    sys.exit(main())
