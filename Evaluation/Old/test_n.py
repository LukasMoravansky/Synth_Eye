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
    image_in = cv2.imread(f'{project_folder}/Data/Camera/Basler/Raw/Image_001.png')

    # Check the image properties
    print("Shape:", image_in.shape)
    print("Data Type:", image_in.dtype)

    """
    # Convert BayerRG8 to RGB
    rgb_image = cv2.cvtColor(image_in, cv2.COLOR_BAYER_RG2RGB)
    
    # Saves the image to the specified file.
    cv2.imwrite('Img_res_1.png', rgb_image)

    print('[INFO] The data processing was completed successfully.')"
    """

if __name__ == '__main__':
    sys.exit(main())