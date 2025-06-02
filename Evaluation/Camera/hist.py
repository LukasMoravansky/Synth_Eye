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
#   ../Utilities/File_IO
import Utilities.File_IO as File_IO

import numpy as np


def main():
    """
    Description:
        A program to adjust the contrast {alpha} and brightness {beta} of a raw image.

        Note:
            The raw images were collected using the script here:
                ./Data_Collection/scan.py
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    image_data = cv2.imread(f'{project_folder}/Data/Camera/Basler/Image_{100:03}.png')
    
    # Function to adjust the contrast and brightness parameters of the input image 
    # by clipping the histogram.
    (alpha_custom, beta_custom) = Utilities.Image_Processing.Get_Alpha_Beta_Parameters(image_data, 1.0)  
    
    # Adjust the contrast and brightness of the image using the alpha and beta parameters.
    #   Equation:
    #       g(i, j) = alpha * f(i, j) + beta
    image_out = cv2.convertScaleAbs(image_data, alpha=alpha_custom, beta=beta_custom)

    # Apply Gaussian blur to the adjusted image
    #img_filtered = cv2.GaussianBlur(image_out, (5, 5), cv2.BORDER_DEFAULT)

    # Saves the image to the specified file.
    cv2.imwrite('Test_100.png', image_out)

    print('[INFO] The data processing was completed successfully.')

if __name__ == '__main__':
    sys.exit(main())