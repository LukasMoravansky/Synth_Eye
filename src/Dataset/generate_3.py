# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' not in sys.path:
    sys.path.append('../')
# OS (Operating system interfaces)
import os
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Utilities/File_IO
import Utilities.File_IO as File_IO
import numpy as np
import Utils

Cls_Id_Remove = [2]
Dataset_Name = 'Dataset_v2'

def main():
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    folder_path = f'{project_folder}/Data/Camera/Basler_a2A1920_51gcPRO_Computar_M1228_MPW3'

    image_data_tmp = cv2.imread(f'{folder_path}/Checkerboard.png')

    if image_data_tmp is None:
        raise FileNotFoundError(f'Unable to load image.')

    image_data = Utils.process_real_image(image_data_tmp)

    cv2.imwrite(f'{folder_path}/Checkerboard_Processed.png', image_data.copy())

if __name__ == '__main__':
    sys.exit(main())