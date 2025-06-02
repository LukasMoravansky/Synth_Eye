import sys
import os
import cv2
import numpy as np

# Add access to your project utilities
if '../../src' not in sys.path:
    sys.path.append('../../src')

import Utilities.Image_Processing
import Utilities.File_IO as File_IO
import Utilities.General

# Constants
CONST_INIT_INDEX = 26

# --- Helper Functions ---

def yolo_to_absolute(x_c, y_c, w, h, img_w, img_h):
    abs_x = x_c * img_w
    abs_y = y_c * img_h
    abs_w = w * img_w
    abs_h = h * img_h
    return abs_x, abs_y, abs_w, abs_h

def absolute_to_yolo(x, y, w, h, img_w, img_h):
    return x / img_w, y / img_h, w / img_w, h / img_h

def yolo_to_pixel_bbox(bbox, img_w, img_h):
    x_c, y_c, w, h = bbox
    x_abs, y_abs, w_abs, h_abs = yolo_to_absolute(x_c, y_c, w, h, img_w, img_h)
    x_min = int(x_abs - w_abs / 2)
    y_min = int(y_abs - h_abs / 2)
    x_max = int(x_abs + w_abs / 2)
    y_max = int(y_abs + h_abs / 2)
    return x_min, y_min, x_max, y_max

def add_random_noise_to_background(image, object_bbox):
    output = image.copy()
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = object_bbox
    mask = np.ones((h, w), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 0
    noise = np.random.randint(0, 256, size=image.shape, dtype=np.uint8)
    output[mask == 1] = noise[mask == 1]
    return output

def add_random_shapes_to_background(image, object_bbox, min_count=10, max_count=100):
    output = image.copy()
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = object_bbox
    count = np.random.randint(min_count, max_count)

    for _ in range(count):
        shape_type = np.random.choice(['line', 'circle', 'rect'])

        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        if x_min <= x1 <= x_max and y_min <= y1 <= y_max:
            continue

        color = tuple(np.random.randint(0, 256, size=3).tolist())
        thickness = np.random.randint(1, 4)

        if shape_type == 'line':
            x2, y2 = x1 + np.random.randint(-100, 100), y1 + np.random.randint(-100, 100)
            if x_min <= x2 <= x_max and y_min <= y2 <= y_max:
                continue
            cv2.line(output, (x1, y1), (x2, y2), color, thickness)
        elif shape_type == 'circle':
            radius = np.random.randint(5, 100)
            cv2.circle(output, (x1, y1), radius, color, -1)
        elif shape_type == 'rect':
            x2, y2 = x1 + np.random.randint(10, 100), y1 + np.random.randint(10, 100)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, -1)
    return output


def mess_up_background(image, object_bbox):
    img = add_random_noise_to_background(image, object_bbox)
    img = add_random_shapes_to_background(img, object_bbox)
    return img

# --- Main Program ---

def main():
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'
    image_path = f'{project_folder}/Data/Dataset_v1/images/train/Image_{CONST_INIT_INDEX:03}.png'
    label_path = f'{project_folder}/Data/Dataset_v1/labels/train/Image_{CONST_INIT_INDEX:03}'

    image_data = cv2.imread(image_path)
    label_data = File_IO.Load(label_path, 'txt', ' ')

    img_h, img_w = image_data.shape[:2]

    object_bbox_yolo = None
    new_labels = []

    # Find object bbox
    for label in label_data:
        if int(label[0]) == 0:
            object_bbox_yolo = label[1:5]
            break

    if object_bbox_yolo is None:
        print("No object bounding box (class 0) found.")
        return

    # Convert object bbox to pixel coordinates
    x_min, y_min, x_max, y_max = yolo_to_pixel_bbox(object_bbox_yolo, img_w, img_h)

    # Crop the object image
    cropped_img = image_data[y_min:y_max, x_min:x_max]
    cropped_h, cropped_w = cropped_img.shape[:2]

    # Process defects
    for label in label_data:
        if int(label[0]) == 2:
            def_x, def_y, def_w, def_h = label[1:5]
            def_x_abs, def_y_abs, def_w_abs, def_h_abs = yolo_to_absolute(def_x, def_y, def_w, def_h, img_w, img_h)
            new_x_abs = def_x_abs - x_min
            new_y_abs = def_y_abs - y_min
            if not (0 <= new_x_abs <= cropped_w and 0 <= new_y_abs <= cropped_h):
                continue
            new_def_x, new_def_y, new_def_w, new_def_h = absolute_to_yolo(
                new_x_abs, new_y_abs, def_w_abs, def_h_abs, cropped_w, cropped_h
            )
            new_labels.append([2, new_def_x, new_def_y, new_def_w, new_def_h])

    # Save cropped image and transformed defect bbox
    cropped_img_path = f'Image_Res_{CONST_INIT_INDEX:03}_Cropped.png'
    cv2.imwrite(cropped_img_path, cropped_img)

    if new_labels:
        lines = [f"{int(lbl[0])} " + ' '.join(f"{v:.6f}" for v in lbl[1:]) for lbl in new_labels]
        File_IO.Save(f'Image_Res_{CONST_INIT_INDEX:03}_Cropped', lines, 'txt', ' ')
        print(f"Saved cropped image and {len(new_labels)} transformed defect label(s).")

    # Generate background mess
    messed_img = mess_up_background(image_data, (x_min, y_min, x_max, y_max))
    messed_img_path = f'Image_Res_{CONST_INIT_INDEX:03}_Messed.png'
    cv2.imwrite(messed_img_path, messed_img)
    print(f"Saved background-messed image: {messed_img_path}")

if __name__ == '__main__':
    sys.exit(main())
