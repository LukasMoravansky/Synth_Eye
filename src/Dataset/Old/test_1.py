import sys
import os
import cv2
import numpy as np

# Add access to the custom utilities
if '../../src' not in sys.path:
    sys.path.append('../../src')

# Custom Utilities
import Utilities.Image_Processing
import Utilities.File_IO as File_IO
import Utilities.General

CONST_INIT_INDEX = 26
CONST_OBJECT_BB_COLOR = [(255, 165, 0), (0, 165, 255), (80, 0, 255)]

def yolo_to_absolute(x_c, y_c, w, h, img_w, img_h):
    abs_x = x_c * img_w
    abs_y = y_c * img_h
    abs_w = w * img_w
    abs_h = h * img_h
    return abs_x, abs_y, abs_w, abs_h

def absolute_to_yolo(x, y, w, h, img_w, img_h):
    return x / img_w, y / img_h, w / img_w, h / img_h

def main():
    # Locate the project folder
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Load the image and label
    image_path = f'{project_folder}/Data/Dataset_v1/images/train/Image_{CONST_INIT_INDEX:03}.png'
    label_path = f'{project_folder}/Data/Dataset_v1/labels/train/Image_{CONST_INIT_INDEX:03}'

    image_data = cv2.imread(image_path)
    label_data = File_IO.Load(label_path, 'txt', ' ')

    img_h, img_w = image_data.shape[:2]

    # Initialize
    object_bbox = None

    # First, find object bbox
    for label in label_data:
        class_id = int(label[0])
        if class_id == 0:
            object_bbox = label[1:5]  # [x_center, y_center, width, height]
            break

    if object_bbox is None:
        print("No object (class 0) bounding box found.")
        return

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

    # Now, transform defect bboxes
    new_labels = []
    for label in label_data:
        class_id = int(label[0])
        if class_id == 2:
            # Defect bbox
            def_x, def_y, def_w, def_h = label[1:5]
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
            new_label = [2, new_def_x, new_def_y, new_def_w, new_def_h]
            new_labels.append(new_label)

    # Save the cropped image
    output_image_path = f'Image_Res_{CONST_INIT_INDEX:03}_Cropped.png'
    cv2.imwrite(output_image_path, cropped_img)

    # Save transformed label(s)
    if new_labels:
        lines = []
        for lbl in new_labels:
            line = f"{int(lbl[0])} " + ' '.join(f"{v:.6f}" for v in lbl[1:])
            lines.append(line)

        File_IO.Save(f'Image_Res_{CONST_INIT_INDEX:03}_Cropped', lines, 'txt', ' ')
        print(f"Saved cropped image and {len(new_labels)} defect label(s).")
    else:
        print("No valid defect labels found inside the object bbox.")

if __name__ == '__main__':
    sys.exit(main())