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

Cls_Id = [2]
Dataset_Name = 'Dataset_v3'

def main():
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    for partition_name in ['train', 'valid']:
        folder_path = f'{project_folder}/Data/Dataset_v1/images/{partition_name}'
        file_info_list = Utils.extract_numbers_from_filenames(folder_path)
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
                image_data_tmp = cv2.imread(file_info['path'])

                if image_data_tmp is None:
                    raise FileNotFoundError(f"Unable to load image from: {file_info['path']}")

                if partition_name in ['train', 'valid']:
                    image_data = Utils.process_synthetic_image(image_data_tmp)
                else:
                    image_data = Utils.process_real_image(image_data_tmp)
                
                img_h, img_w = image_data.shape[:2]

                defect_bbox = []
                if np.isin(Cls_Id, label_data_tmp[:, 0]).any() and label_data_tmp[:, 0].size >= 1:
                    for _, label_data_i in enumerate(label_data_tmp):
                        if label_data_i[0] in Cls_Id:
                            defect_bbox.append(np.append([0], label_data_i[1:]))
                            break
                else:
                    if label_data_tmp[0, 0] == 0:
                        with open(f'{project_folder}/Data/{Dataset_Name}/labels/{partition_name}/{file_info["name"]}.txt', 'w') as f:
                            pass
                    else:
                        continue

                # Convert object bbox to pixel coordinates
                obj_x, obj_y, obj_w, obj_h = object_bbox
                obj_x_abs, obj_y_abs, obj_w_abs, obj_h_abs = Utils.yolo_to_absolute(obj_x, obj_y, obj_w, obj_h, img_w, img_h)

                obj_left = int(obj_x_abs - obj_w_abs / 2)
                obj_top = int(obj_y_abs - obj_h_abs / 2)
                obj_right = int(obj_x_abs + obj_w_abs / 2)
                obj_bottom = int(obj_y_abs + obj_h_abs / 2)

                # Crop the object region from the original image
                cropped_img = image_data[obj_top:obj_bottom, obj_left:obj_right]
                cropped_h, cropped_w = cropped_img.shape[:2]

                if len(defect_bbox) > 0:
                    # Defect bbox
                    def_x, def_y, def_w, def_h = defect_bbox[0][1::]
                    def_x_abs, def_y_abs, def_w_abs, def_h_abs = Utils.yolo_to_absolute(def_x, def_y, def_w, def_h, img_w, img_h)

                    # Shift defect coordinates relative to cropped image
                    new_x_abs = def_x_abs - obj_left
                    new_y_abs = def_y_abs - obj_top

                    # Skip if defect center is outside cropped image
                    if not (0 <= new_x_abs <= cropped_w and 0 <= new_y_abs <= cropped_h):
                        continue

                    # Convert to YOLO relative to cropped image
                    new_def_x, new_def_y, new_def_w, new_def_h = Utils.absolute_to_yolo(
                        new_x_abs, new_y_abs, def_w_abs, def_h_abs, cropped_w, cropped_h
                    )

                    # Clamp values to [0, 1] if needed
                    new_label = np.array([[0, new_def_x, new_def_y, new_def_w, new_def_h]], dtype=label_data_tmp.dtype)

                    new_label[:, 0] = new_label[:, 0].astype(int)
                    for _, new_label_i in enumerate(new_label):
                        formatted_data = f'{int(new_label_i[0])} ' + ' '.join(f'{x:.6f}' for x in new_label_i[1:])
                        File_IO.Save(f'{project_folder}/Data/{Dataset_Name}/labels/{partition_name}/{file_info["name"]}', formatted_data.split(), 'txt', ' ')

            cv2.imwrite(f'{project_folder}/Data/{Dataset_Name}/images/{partition_name}/{file_info["name"]}.png', cropped_img.copy())
            print(f'{partition_name} | {file_info["name"]}')
    
if __name__ == '__main__':
    sys.exit(main())