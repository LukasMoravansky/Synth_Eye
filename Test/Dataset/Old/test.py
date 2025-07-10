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

import Utilities.General
import numpy as np

"""
Description:
    Initialization of constants.
"""
# The color of the bounding box of the object.
CONST_OBJECT_BB_COLOR = [(255, 165, 0), (0, 165, 255), (80, 0, 255)]

def main():
    """
    Description:
        A program to evaluate synthetic data (image with corresponding label) generated from Blender.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    for idx in range(4501, 5001):
        # Load a raw image from a file.
        image_data = cv2.imread(f'{project_folder}/Data/Dataset_v1/images/valid/Image_{idx:03}.png')
        # Load a label (annotation) from a file.
        label_data = File_IO.Load(f'{project_folder}/Data/Dataset_v1/labels/valid/Image_{idx:03}', 'txt', ' ')

        for i, label_data_i in enumerate(label_data):
            # Create a bounding box from the label data.
            Bounding_Box_Properties = {'Name': f'{int(label_data_i[0])}_{i}', 'Precision': None, 
                                       'Data': {'x_c': label_data_i[1], 'y_c': label_data_i[2], 'width': label_data_i[3], 'height': label_data_i[4]}}

            # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
            # the raw image.
            image_data = Utilities.Image_Processing.Draw_Bounding_Box(image_data, Bounding_Box_Properties, 'YOLO', CONST_OBJECT_BB_COLOR[int(label_data_i[0])], 
                                                                      True, False)

        # Saves the images to the specified file.
        cv2.imwrite(f'{project_folder}/Data/Dataset_v2/Image_{idx:03}.png', image_data)
    
if __name__ == '__main__':
    sys.exit(main())