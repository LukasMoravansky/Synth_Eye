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
#   ../Parameters/Scene
import Parameters.Scene
#   ../Calibration/Core
import Calibration.Core

"""
Description:
    Initialization of constants.
"""
# The identification number of the iteration to load the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 1

def main():
    """
    Description:
        A program to perform camera calibration using images generated synthetically from a virtual camera setup.  
        Each image is loaded from a file produced by Blender, simulating a Basler a2A1920-51gcPRO camera equipped with  
        virtual EFFI-FD-200-200-000 lighting to ensure realistic illumination.

        The calibration process computes the camera matrix, distortion coefficients, and pixel-to-millimeter conversion factors 
        to correct synthetic image distortions and align the virtual camera model with real-world camera parameters.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Define input and output image path.
    input_path = os.path.join(f'{project_folder}/Data/Camera/{Parameters.Scene.Basler_Cam_Str.Name}_Virtual/', 
                              f'Image_Checkerboard_{(CONST_INIT_INDEX):03}.png')
    output_path = os.path.join(f'{project_folder}/Data/Camera/{Parameters.Scene.Basler_Cam_Str.Name}_Virtual/', 
                              f'Image_Checkerboard_{(CONST_INIT_INDEX):03}_Processed.png')

    # Load a raw image from a file.
    img_raw = cv2.imread(input_path)

    # Initialize the class for custom image processing.
    Process_Image_Cls = Utilities.Image_Processing.Process_Image_Cls('synthetic')

    # Apply the image processing pipeline.
    img_raw_processed = Process_Image_Cls.Apply(img_raw)

    # Initialize the checkerboard calibration class with checkerboard dimensions and square size (in mm).
    Checkerboard_Calib_Cls = Calibration.Core.Checkerboard_Calibration_Cls(inner_corners=(11, 8), square_size=12.0)

    # Perform camera calibration using the provided checkerboard image.
    flag, x = Checkerboard_Calib_Cls.Solve(img_raw_processed, False, f'{project_folder}/Data/Camera/{Parameters.Scene.Basler_Cam_Str.Name}_Virtual')

    # Check whether calibration was successful.
    if flag == True:
        # Display the camera calibration results: intrinsic matrix, distortion coefficients, and scaling factor.
        print(f'[INFO] Intrinsic Matrix (K):\n{x.K}')
        print(f'[INFO] Distortion Coefficients:\n{x.Coefficients}')
        print(f'[INFO] Pixel-to-mm Conversion Factor:\n{x.Conversion_Factor}')

        # Save the image after successful calibration to the specified output path.
        cv2.imwrite(output_path, img_raw_processed)
        print('[INFO] The data processing was completed successfully.')
        print(f'[INFO] Output image saved to: {output_path}')
    else:
        # Inform user that checkerboard detection or calibration failed.
        print('[WARNING] Data processing was not completed successfully. An error occurred.')

    # Release the classes.
    del Process_Image_Cls

if __name__ == '__main__':
    sys.exit(main())