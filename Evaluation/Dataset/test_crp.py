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

CONST_INIT_INDEX = 9
CONST_OBJECT_BB_COLOR = [(255, 165, 0), (0, 165, 255), (80, 0, 255)]

def convert_cropped_yolo_to_original(x_c, y_c, w, h, cropped_w, cropped_h, obj_left, obj_top, orig_w, orig_h):
    # Convert YOLO (relative) back to absolute coordinates in the cropped image
    abs_x = x_c * cropped_w
    abs_y = y_c * cropped_h
    abs_w = w * cropped_w
    abs_h = h * cropped_h

    # Shift back to original image coordinates
    abs_x_orig = abs_x + obj_left
    abs_y_orig = abs_y + obj_top

    # Convert absolute to YOLO in original image
    return absolute_to_yolo(abs_x_orig, abs_y_orig, abs_w, abs_h, orig_w, orig_h)

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
    project_folder = os.getcwd().split('Synth_Eye_Old')[0] + 'Synth_Eye_Old'

    # Load the image and label
    image_path = f'{project_folder}/Data/Dataset_v1/images/train/Image_{CONST_INIT_INDEX:03}.png'
    label_path = f'{project_folder}/Data/Dataset_v1/labels/train/Image_{CONST_INIT_INDEX:03}'

    image_data = cv2.imread(image_path)
    label_data = File_IO.Load(label_path, 'txt', ' ')

    img_h, img_w = image_data.shape[:2]

    cv2.imwrite(f'Image_Res_{CONST_INIT_INDEX:03}_Original.png', image_data)

    """
    General object detection	imgsz=640 or imgsz=960
    Higher accuracy, larger objects	imgsz=1280
    """
    resized_image = cv2.resize(image_data, (960, 960))
    cv2.imwrite(f'Image_Res_{CONST_INIT_INDEX:03}_Original_Resized.png', resized_image)

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

            Bounding_Box_Properties = {'Name': f'{int(lbl[0])}', 'Precision': None, 
                                       'Data': {'x_c': lbl[1], 'y_c': lbl[2], 'width': lbl[3], 'height': lbl[4]}}

            cropped_img_bb = Utilities.Image_Processing.Draw_Bounding_Box(cropped_img, Bounding_Box_Properties, 'YOLO', CONST_OBJECT_BB_COLOR[int(lbl[0])], 
                                                                True, False)

        cv2.imwrite(f'Image_Res_{CONST_INIT_INDEX:03}_Cropped_BB.png', cropped_img_bb)

        #File_IO.Save(f'Image_Res_{CONST_INIT_INDEX:03}_Cropped', lines, 'txt', ' ')
        print(f"Saved cropped image and {len(new_labels)} defect label(s).")
    else:
        print("No valid defect labels found inside the object bbox.")

    # Reverse-transform: map cropped YOLO back to original image YOLO
    orig_def_x, orig_def_y, orig_def_w, orig_def_h = convert_cropped_yolo_to_original(
        new_def_x, new_def_y, new_def_w, new_def_h,
        cropped_w, cropped_h, obj_left, obj_top,
        img_w, img_h
    )

    print(new_def_x, new_def_y, new_def_w, new_def_h)
    print(cropped_w, cropped_h, cropped_img.shape)
    print(obj_left, obj_top)
    print(img_w, img_h)
    # (Optional) Store or draw it back on original image
    reversed_label = [2, orig_def_x, orig_def_y, orig_def_w, orig_def_h]

    Bounding_Box_Properties_Orig = {'Name': f'{int(reversed_label[0])}', 'Precision': None,
                                'Data': {'x_c': reversed_label[1], 'y_c': reversed_label[2], 'width': reversed_label[3], 'height': reversed_label[4]}}

    image_data_bb = Utilities.Image_Processing.Draw_Bounding_Box(image_data, Bounding_Box_Properties_Orig, 'YOLO', CONST_OBJECT_BB_COLOR[int(reversed_label[0])], True, False)
    cv2.imwrite(f'Image_Res_{CONST_INIT_INDEX:03}_Original_With_Reversed_BB.png', image_data_bb)


if __name__ == '__main__':
    sys.exit(main())
