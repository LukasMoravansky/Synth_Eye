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
from pathlib import Path
import re

Cls_Id_Remove = [2]
Dataset_Name = 'Dataset_v1a'

def extract_numbers_from_filenames(folder_path):
    folder = Path(folder_path)
    results = []
    
    for file in folder.iterdir():
        if file.is_file() and file.suffix == '.png':
            match = re.search(r'Image_((?:\w+_)*)(\d+)\.png', file.name)
            if match:
                prefix = match.group(1)
                number = int(match.group(2))
                is_background = 'Background' in prefix
                results.append({
                    'name': file.name,
                    'id': number,
                    'is_background': is_background,
                    'path': str(file.with_suffix(''))
                })
    
    return results

def main():
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # 'train', 'valid', 
    for partition_name in ['train', 'valid', 'test']:
        folder_path = f'{project_folder}/Data/Dataset_v1/images/{partition_name}'
        file_info_list = extract_numbers_from_filenames(folder_path)
        for file_info in file_info_list:
            print(file_info)
            #print(f'Image_{i:03}')

            """
            # read data... 
            image_data = cv2.imread(f'{project_folder}/Data/Dataset_v1/images/{partition_name_i}/Image_{i:03}.png')
            label_data_tmp = File_IO.Load(f'{project_folder}/Data/Dataset_v1/labels/{partition_name_i}/Image_{i:03}', 'txt', ' ')

            # processing...
            label_data = []
            if np.isin(Cls_Id_Remove, label_data_tmp[:, 0]).any() and label_data_tmp[:, 0].size >= 1:
                for _, label_data_i in enumerate(label_data_tmp):
                    if label_data_i[0] not in Cls_Id_Remove:
                        label_data.append(label_data_i)
                
                label_data = np.array(label_data, dtype=label_data_i.dtype)
            else:
                if not np.isin(Cls_Id_Remove, label_data_tmp[0, 0]).any():
                    label_data = label_data_tmp.copy()
                else:
                    label_data = np.array(label_data, dtype=label_data_tmp.dtype)

            #   Label
            if label_data.size > 0:
                label_data[:, 0] = label_data[:, 0].astype(int)
                for _, label_data_i in enumerate(label_data):
                    formatted_data = f'{int(label_data_i[0])} ' + ' '.join(f'{x:.6f}' for x in label_data_i[1:])
                    File_IO.Save(f'{project_folder}/Data/{Dataset_Name}/labels/{partition_name_i}/Image_{i}', formatted_data.split(), 'txt', ' ')
            else:
                with open(f'{project_folder}/Data/{Dataset_Name}/labels/{partition_name_i}/Image_{i}.txt', 'w') as f:
                    pass

            #   Image
            cv2.imwrite(f'{project_folder}/Data/{Dataset_Name}/images/{partition_name_i}/Image_{i:03}.png', image_data.copy())
            """
    
if __name__ == '__main__':
    sys.exit(main())