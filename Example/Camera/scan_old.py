# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OpenCV library for computer vision tasks
import cv2
# Numpy (Array computing)
import numpy as np
# OS module for file handling and accessing directories
import os
# Custom Lib.:
#   ../Basler/Camera
from Basler.Camera import Basler_Cls
#   ../Parameters/Scene
import Parameters.Scene
#   ../Calibration/Parameters
from Calibration.Parameters import Basler_Calib_Param_Str
#   ../Utilities/Image_Processing
import Utilities.Image_Processing

"""
Description:
    Initialization of constants.
"""
# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 100

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
        raise ValueError('[ERROR] No image captured!')

    # Release the camera resources.
    del Basler_Cam_Id_1

    # Define output image path.
    output_path = os.path.join(f'{project_folder}/Data/Camera/{Parameters.Scene.Basler_Cam_Str.Name}/', f'Image_{(CONST_INIT_INDEX):03}.png')

    # Initialize the class for custom image processing.
    Process_Image_Cls = Utilities.Image_Processing.Process_Image_Cls('real')

    # Apply the image processing pipeline.
    img_raw_processed = Process_Image_Cls.Apply(img_raw)

    # Undistort the image using camera calibration parameters.
    h, w = img_raw_processed.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(Basler_Calib_Param_Str.K, np.array(list(Basler_Calib_Param_Str.Coefficients.values()), dtype=np.float64), 
                                                         (w, h), 1, (w, h))
    img_undistorted = cv2.undistort(img_raw_processed, Basler_Calib_Param_Str.K, np.array(list(Basler_Calib_Param_Str.Coefficients.values()), dtype=np.float64), 
                                    None, new_camera_matrix)

    # Save the image with bounding boxes.
    cv2.imwrite(output_path, img_undistorted)
    print(f'[INFO] Result saved at: {output_path}')

    # Release the classes.
    del Process_Image_Cls

if __name__ == '__main__':
    sys.exit(main())