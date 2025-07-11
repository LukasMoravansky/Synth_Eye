# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OS (Operating system interfaces)
import os
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Utilities/Image_Processing
import Utilities.Image_Processing
#   ../Basler/Camera
from Basler.Camera import Basler_Cls
#   ../Parameters/Scene
import Parameters.Scene

"""
Description:
    Initialization of constants.
"""
# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 1

def main():
    """
    Description:
        A program to capture images of a checkerboard pattern using a Basler a2A1920-51gcPRO camera and perform camera calibration based 
        on the captured images. The system is equipped with EFFI-FD-200-200-000 lighting to ensure optimal illumination.

        The calibration computes the camera matrix, distortion coefficients, and pixel-to-millimeter conversion factors 
        to correct image distortions.

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
    output_path = os.path.join(f'{project_folder}/Data/Camera/{Parameters.Scene.Basler_Cam_Str.Name}/', f'Image_Checkerboard_{(CONST_INIT_INDEX):03}.png')

    # Initialize the class for custom image processing.
    Process_Image_Cls = Utilities.Image_Processing.Process_Image_Cls('real')

    # Apply the image processing pipeline.
    img_raw_processed = Process_Image_Cls.Apply(img_raw)

    # Function to adjust the contrast and brightness parameters of the input image 
    # by clipping the histogram.
    (alpha_custom, beta_custom) = Utilities.Image_Processing.Get_Alpha_Beta_Parameters(img_raw_processed, 1.0)  
    
    # Adjust the contrast and brightness of the image using the alpha and beta parameters.
    #   Equation:
    #       g(i, j) = alpha * f(i, j) + beta
    image_out = cv2.convertScaleAbs(img_raw_processed, alpha=alpha_custom, beta=beta_custom)

    # Saves the image to the specified file.
    cv2.imwrite(output_path, image_out)
    print('[INFO] The data processing was completed successfully.')

    # Release the classes.
    del Process_Image_Cls

if __name__ == '__main__':
    sys.exit(main())