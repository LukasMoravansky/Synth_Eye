# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OS (Operating system interfaces)
import os
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Time (Time access and conversions)
import time
# YAML (configuration file parsing)
import yaml
# Ultralytics (Real-time object detection and image segmentation model)
from ultralytics import YOLO
# PyTorch (tensors and dynamic neural networks)
import torch
# Custom Library:
#   ../Utilities/Image_Processing
import Utilities.Image_Processing
#   ../Utilities/General
import Utilities.General

"""
Description:
    Initialization of constants.
"""
# The name of the dataset, model, and color of the object bounding boxes.
CONST_CONFIG_MODEL_OBJ = {'Model': 'yolov8m_object_detection', 'Color': [(255, 165, 0), (0, 165, 255)]}
CONST_CONFIG_MODEL_DEFECT = {'Model': 'yolov8m_defect_detection', 'Color': [(80, 0, 255)]}

def main():
    """
    Description:
        A program to perform AI-based object detection and defect detection
        on a test image dataset using two pre-trained YOLOv8 models.

        1. Load test images.
        2. Detect objects with the first YOLO model.
        3. Crop detected objects.
        4. Detect defects inside cropped objects with the second YOLO model.
        5. Draw bounding boxes and save the images.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Detect and assign the training device (GPU if available).
    device_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device_id}")
    if device_id.type == 'cuda':
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

    # Load hyperparameters and dataset configuration files.
    with open(os.path.join(project_folder, 'Training', 'Args_Model_1.yaml'), 'r') as f:
        meta_args = yaml.safe_load(f)

    # Load a pre-trained custom YOLO models.
    model_object = YOLO(f"{project_folder}/YOLO/Model/Dataset_v2/{CONST_CONFIG_MODEL_OBJ['Model']}.pt")
    model_defect = YOLO(f"{project_folder}/YOLO/Model/Dataset_v3/{CONST_CONFIG_MODEL_DEFECT['Model']}.pt")

    # Extract numeric identifiers and background flags from filenames in a specified directory.
    image_dir_info = Utilities.General.Extract_Num_From_Filename('Image', os.path.join(project_folder, 'Data', 'Dataset_v2', 'images', 'test'))

    # Initialize the timer and counters.
    start_time = time.time(); total_processed = 0

    for image_dir_info_i in image_dir_info:

        image_path = image_dir_info_i['Path']; image_name = image_dir_info_i['Name']

        # Load image.
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'[ERROR] Unable to load image from: {image_path}')

        # ---------------------------------------------------------------------
        # Stage 1: Object Detection
        # ---------------------------------------------------------------------
        results_object = model_object.predict(source=image_path, device=device_id, imgsz=meta_args['imgsz'], conf=0.25, iou=0.5)

        if results_object[0].boxes.shape[0] >= 1:

            class_id = results_object[0].boxes.cls.cpu().numpy(); b_box_obj = results_object[0].boxes.xywhn.cpu().numpy(); conf = results_object[0].boxes.conf.cpu().numpy()

            for _, (class_id_i, b_box_i, conf_i) in enumerate(zip(class_id, b_box_obj, conf)):

                # Draw object bounding box
                Bounding_Box_Properties = {'Name': f'{int(class_id_i)}', 'Precision': f'{str(conf_i)[0:5]}', 
                                           'Data': {'x_c': b_box_i[0], 'y_c': b_box_i[1], 'width': b_box_i[2], 'height': b_box_i[3]}}
                
                processed_image = Utilities.Image_Processing.Draw_Bounding_Box(image, Bounding_Box_Properties, 'YOLO', CONST_CONFIG_MODEL_OBJ['Color'][int(class_id_i)], True, True)
                image = processed_image.copy()

                # -----------------------------------------------------------------
                # Crop the object from the image
                # -----------------------------------------------------------------
                img_h, img_w = image.shape[:2]
                Resolution = {'x': img_w, 'y': img_h}
                abs_coordinates_obj = Utilities.General.YOLO_To_Absolute_Coordinates({'x_c': b_box_i[0], 'y_c': b_box_i[1], 'width': b_box_i[2], 'height': b_box_i[3]}, Resolution)
                obj_left = int(abs_coordinates_obj['x'] - abs_coordinates_obj['width']/2)
                obj_top = int(abs_coordinates_obj['y'] - abs_coordinates_obj['height']/2)
                obj_right = int(abs_coordinates_obj['x'] + abs_coordinates_obj['width']/2)
                obj_bottom = int(abs_coordinates_obj['y'] + abs_coordinates_obj['height']/2)
                obj_left = max(0, obj_left); obj_top = max(0, obj_top); obj_right = min(img_w, obj_right); obj_bottom = min(img_h, obj_bottom)
                cropped_obj = image[obj_top:obj_bottom, obj_left:obj_right]

                if cropped_obj.size == 0:
                    continue

                # -----------------------------------------------------------------
                # Stage 2: Defect Detection on cropped object
                # -----------------------------------------------------------------
                results_defect = model_defect.predict(source=cropped_obj, device=device_id, imgsz=meta_args['imgsz'], conf=0.25, iou=0.5)

                if results_defect[0].boxes.shape[0] >= 1:

                    defect_cls = results_defect[0].boxes.cls.cpu().numpy(); defect_b_box = results_defect[0].boxes.xywhn.cpu().numpy(); defect_conf = results_defect[0].boxes.conf.cpu().numpy()

                    for _, (d_class_i, d_b_box_i, d_conf_i) in enumerate(zip(defect_cls, defect_b_box, defect_conf)):

                        resolution_crop = {'x': cropped_obj.shape[1], 'y': cropped_obj.shape[0]}
                        abs_defect = Utilities.General.YOLO_To_Absolute_Coordinates({'x_c': d_b_box_i[0], 'y_c': d_b_box_i[1], 'width': d_b_box_i[2], 'height': d_b_box_i[3]}, resolution_crop)
                        abs_defect['x'] += obj_left; abs_defect['y'] += obj_top

                        Bounding_Box_Defect = {'Name': f'Defect-{int(d_class_i)}', 'Precision': f'{str(d_conf_i)[0:5]}', 'Data': abs_defect}
                        processed_image = Utilities.Image_Processing.Draw_Bounding_Box(image, Bounding_Box_Defect, 'Absolute', CONST_CONFIG_MODEL_DEFECT['Color'][0], True, True)
                        image = processed_image.copy()

        else:
            processed_image = image.copy()

        # Save processed image.
        cv2.imwrite(f"{project_folder}/YOLO/Prediction/Dataset_v2_ALL/{image_name}.png", processed_image)

        # Tracking the number of predictions.
        total_processed += 1

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print('[INFO] Data generation completed successfully.')
    print(f'[INFO] Total samples processed: {total_processed}')
    print(f'[INFO] Time: {int(minutes)}m {int(seconds)}s')


if __name__ == '__main__':
    sys.exit(main())
