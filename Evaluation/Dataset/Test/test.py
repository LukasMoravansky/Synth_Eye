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
# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 26
# The color of the bounding box of the object.
CONST_OBJECT_BB_COLOR = [(255, 165, 0), (0, 165, 255), (80, 0, 255)]

def main():
    """
    Description:
        A program to evaluate synthetic data (image with corresponding label) generated from Blender.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Load a raw image from a file.
    image_data = cv2.imread(f'{project_folder}/Data/Dataset_v1/images/train/Image_{CONST_INIT_INDEX:03}.png')
    # Load a label (annotation) from a file.
    label_data = File_IO.Load(f'{project_folder}/Data/Dataset_v1/labels/train/Image_{CONST_INIT_INDEX:03}', 'txt', ' ')

    for _, label_data_i in enumerate(label_data):
        if int(label_data_i[0]) == 2:
            formatted_data = f'{int(label_data_i[0])} ' + ' '.join(f'{x:.6f}' for x in label_data_i[1::])
            File_IO.Save(f'Image_Res_{CONST_INIT_INDEX:03}_Cropped', formatted_data.split(), 'txt', ' ')
            continue

        data = Utilities.General.Convert_Boundig_Box_Data('YOLO', 'PASCAL_VOC', {'x_c': label_data_i[1], 'y_c': label_data_i[2], 
                                                                                 'width': label_data_i[3], 'height': label_data_i[4]}, 
                                                                                 {'x': image_data.shape[1], 'y': image_data.shape[0]})
        x_min = data['x_min']; y_min = data['y_min']
        x_max = data['x_max']; y_max = data['y_max']

        img_cropped = image_data[y_min:y_max, x_min:x_max]

    # Saves the images to the specified file.
    cv2.imwrite(f'Image_Res_{CONST_INIT_INDEX:03}_Cropped.png', img_cropped)
    
if __name__ == '__main__':
    sys.exit(main())