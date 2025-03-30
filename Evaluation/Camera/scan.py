# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OpenCV library for computer vision tasks
import cv2
# OS module for file handling and accessing directories
import os
# Library to work with Basler cameras
from Basler.Camera import Basler_Cls

# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 99

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
    project_folder = os.getcwd().split('INTEMAC_Synth_Eye')[0] + 'INTEMAC_Synth_Eye'

    # Custom camera configuration.
    custom_cfg = {
        'exposure_time': 1000,
        'gain': 10,
        'balance_ratios': {'Red': 1.1, 'Green': 1.0, 'Blue': 1.3},
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
    output_path = os.path.join(f'{project_folder}/Data/Camera/Basler/Raw/', f'Image_{(CONST_INIT_INDEX):03}.png')

    # Save the image with bounding boxes.
    cv2.imwrite(output_path, img_raw)

    print(f'Result saved at: {output_path}')

if __name__ == '__main__':
    sys.exit(main())