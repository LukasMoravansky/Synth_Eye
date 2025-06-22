import cv2
import numpy as np
import sys
import os
import numpy as np
import Utils

SRC_PATH = os.path.abspath('../../src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from pathlib import Path
import re

def main():
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    counter = 0
    for partition_name in ['test', 'train', 'valid']:
        folder_path = f'{project_folder}/Data/Dataset_v1/images/{partition_name}'
        file_info_list = Utils.extract_numbers_from_filenames(folder_path)
        for file_info in file_info_list:
            counter += 1

    print(counter)
 
if __name__ == '__main__':
    main()