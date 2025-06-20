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

    for partition_name in ['train', 'test', 'valid']:
        folder_path = f'{project_folder}/Data/Dataset_v1/images/{partition_name}'
        file_info_list = Utils.extract_numbers_from_filenames(folder_path)
        for file_info in file_info_list:
            image_data_tmp = cv2.imread(file_info['path'])

            if image_data_tmp is None:
                raise FileNotFoundError(f"Unable to load image from: {file_info['path']}")

            if partition_name in ['train', 'valid']:
                image_data = Utils.process_synthetic_image(image_data_tmp)
            else:
                image_data = image_data_tmp.copy()

            cv2.imwrite(f'{project_folder}/Data/{Dataset_Name}/images/{partition_name}/{file_info["name"]}.png', image_data.copy())

            if file_info['is_background'] == True:
                with open(f'{project_folder}/Data/{Dataset_Name}/labels/{partition_name}/{file_info["name"]}.txt', 'w') as f:
                    pass
            else:
                label_data_tmp = File_IO.Load(f'{project_folder}/Data/Dataset_v1/labels/{partition_name}/{file_info["name"]}', 'txt', ' ')
                
                label_data = []
                if np.isin(Cls_Id_Remove, label_data_tmp[:, 0]).any() and label_data_tmp[:, 0].size >= 1:
                    for _, label_data_i in enumerate(label_data_tmp):
                        if label_data_i[0] not in Cls_Id_Remove:
                            label_data.append(np.append([2], label_data_i[1:]))
                    
                    label_data = np.array(label_data, dtype=label_data_i.dtype)
                else:
                    id_new = 1 if label_data_tmp[0, 0] == 1 else 0
                    label_data = np.array([np.append([id_new], label_data_tmp[0, 1::])], dtype=label_data_tmp.dtype)

                label_data[:, 0] = label_data[:, 0].astype(int)
                for _, label_data_i in enumerate(label_data):
                    formatted_data = f'{int(label_data_i[0])} ' + ' '.join(f'{x:.6f}' for x in label_data_i[1:])
                    File_IO.Save(f'{project_folder}/Data/{Dataset_Name}/labels/{partition_name}/{file_info["name"]}', formatted_data.split(), 'txt', ' ')
    
            print(f'{partition_name} | {file_info["name"]}')
if __name__ == '__main__':
    sys.exit(main())