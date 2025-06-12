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

Cls_Id = [2]
Dataset_Name = 'Dataset_v2'

def yolo_to_absolute(x_c, y_c, w, h, img_w, img_h):
    abs_x = x_c * img_w
    abs_y = y_c * img_h
    abs_w = w * img_w
    abs_h = h * img_h
    return abs_x, abs_y, abs_w, abs_h

def absolute_to_yolo(x, y, w, h, img_w, img_h):
    return x / img_w, y / img_h, w / img_w, h / img_h

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
                    'name': file.stem,
                    'id': number,
                    'is_background': is_background,
                    'path': str(file)
                })
    
    return results

def main():
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    for partition_name in ['test']:
        folder_path = f'{project_folder}/Data/Dataset_v1/images/{partition_name}'
        file_info_list = extract_numbers_from_filenames(folder_path)
        for file_info in file_info_list:
            if file_info['is_background'] == True:
                continue
            else:
                label_data_tmp = File_IO.Load(f'{project_folder}/Data/Dataset_v1/labels/{partition_name}/{file_info["name"]}', 'txt', ' ')

                # First, find object bbox
                for label in label_data_tmp:
                    class_id = int(label[0])
                    if class_id == 0:
                        object_bbox = label[1:5]  # [x_center, y_center, width, height]
                        break

                if object_bbox is None:
                    print("No object (class 0) bounding box found.")
                    return
    
                # ..
                image_data = cv2.imread(file_info['path'])
                img_h, img_w = image_data.shape[:2]

                defect_bbox = np.zeros(4)
                if np.isin(Cls_Id, label_data_tmp[:, 0]).any() and label_data_tmp[:, 0].size >= 1:
                    for _, label_data_i in enumerate(label_data_tmp):
                        if label_data_i[0] in Cls_Id:
                            defect_bbox = np.append([0], label_data_i[1:])
                            break
                else:
                    if label_data_tmp[0, 0] == 0:
                        with open(f'{project_folder}/Data/{Dataset_Name}/labels/{partition_name}/{file_info["name"]}.txt', 'w') as f:
                            pass
                    else:
                        continue

                # Convert object bbox to pixel coordinates
                obj_x, obj_y, obj_w, obj_h = object_bbox
                obj_x_abs, obj_y_abs, obj_w_abs, obj_h_abs = yolo_to_absolute(obj_x, obj_y, obj_w, obj_h, img_w, img_h)

                obj_left = int(obj_x_abs - obj_w_abs / 2)
                obj_top = int(obj_y_abs - obj_h_abs / 2)
                obj_right = int(obj_x_abs + obj_w_abs / 2)
                obj_bottom = int(obj_y_abs + obj_h_abs / 2)

                # Crop the object region from the original image
                cropped_img = image_data[obj_top:obj_bottom, obj_left:obj_right]
                cropped_h, cropped_w = cropped_img.shape[:2]

                if defect_bbox.shape[0] > 0:
                    # Defect bbox
                    def_x, def_y, def_w, def_h = defect_bbox[1:5]
                    def_x_abs, def_y_abs, def_w_abs, def_h_abs = yolo_to_absolute(def_x, def_y, def_w, def_h, img_w, img_h)

                    # Shift defect coordinates relative to cropped image
                    new_x_abs = def_x_abs - obj_left
                    new_y_abs = def_y_abs - obj_top

                    # Skip if defect center is outside cropped image
                    if not (0 <= new_x_abs <= cropped_w and 0 <= new_y_abs <= cropped_h):
                        continue

                    # Convert to YOLO relative to cropped image
                    new_def_x, new_def_y, new_def_w, new_def_h = absolute_to_yolo(
                        new_x_abs, new_y_abs, def_w_abs, def_h_abs, cropped_w, cropped_h
                    )

                    # Clamp values to [0, 1] if needed
                    new_label = np.array([[0, new_def_x, new_def_y, new_def_w, new_def_h]], dtype=label_data_tmp.dtype)

                    new_label[:, 0] = new_label[:, 0].astype(int)
                    for _, new_label_i in enumerate(new_label):
                        formatted_data = f'{int(new_label_i[0])} ' + ' '.join(f'{x:.6f}' for x in new_label_i[1:])
                        File_IO.Save(f'{project_folder}/Data/{Dataset_Name}/labels/{partition_name}/{file_info["name"]}', formatted_data.split(), 'txt', ' ')

            cv2.imwrite(f'{project_folder}/Data/{Dataset_Name}/images/{partition_name}/{file_info["name"]}.png', cropped_img.copy())
    
if __name__ == '__main__':
    sys.exit(main())