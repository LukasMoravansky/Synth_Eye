# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# Time (Time access and conversions)
import time
# OS (Operating system interfaces)
import os
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Utilities/Image_Processing
import Utilities.Image_Processing

"""
Description:
    Initialization of constants.
"""
def main():
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('INTEMAC_Synth_Eye')[0] + 'INTEMAC_Synth_Eye'

    # Loads the image to the specified file.
    image_in = cv2.imread(f'{project_folder}/Data/Camera/raw/images/Image_2.png')
    
    # Function to adjust the contrast and brightness parameters of the input image 
    # by clipping the histogram.
    (alpha_custom, beta_custom) = Utilities.Image_Processing.Get_Alpha_Beta_Parameters(image_in, 0.25)  
    
    # Adjust the contrast and brightness of the image using the alpha and beta parameters.
    #   Equation:
    #       g(i, j) = alpha * f(i, j) + beta
    image_out = cv2.convertScaleAbs(image_in, alpha=alpha_custom, beta=beta_custom)

    # Saves the image to the specified file.
    cv2.imwrite('Img_res_1.png', image_in)
    cv2.imwrite('Img_res_1.png', image_out)


    print('[INFO] The data processing was completed successfully.')

if __name__ == '__main__':
    sys.exit(main())