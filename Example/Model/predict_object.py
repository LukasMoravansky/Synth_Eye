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
# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
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
# The boundaries of an object (bounding box) determined using gen_obj_boundaries.py script.
CONST_OBJECT_BB_AREA = {'Min': 0.1, 'Max':0.15}

def main():
    """
    Description:
        A program to perform AI-based object and defect detection on a test image dataset using a pre-trained 
        YOLOv8 model.

        1. Model: Object detection - yolov8m_object_detection.pt
        2. Model: Defect detection - yolov8m_defect_detection.pot

        It loads test images, runs inference, draws bounding boxes around detected 
        objects/defects, and saves the annotated images.

        Note:
            For more information about the training process, see: {project_folder}/Training
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

    # Load a pre-trained custom YOLO model.
    model_object = YOLO(f"{project_folder}/YOLO/Model/Dataset_v2/{CONST_CONFIG_MODEL_OBJ['Model']}.pt")
    model_defect = YOLO(f"{project_folder}/YOLO/Model/Dataset_v3/{CONST_CONFIG_MODEL_DEFECT['Model']}.pt")

    # Extracts numeric identifiers and background flags from filenames in a specified directory.
    image_dir_info = Utilities.General.Extract_Num_From_Filename('Image', os.path.join(project_folder, 'Data', 'Dataset_v2', 
                                                                                       'images', 'test'))

    # Initialize the timer and counters.
    start_time = time.time(); total_processed = 0

    for image_dir_info_i in image_dir_info:
        image_path = image_dir_info_i['Path']; image_name = image_dir_info_i['Name']

        # Load image.
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'[ERROR] Unable to load image from: {image_path}')

        # Perform prediction of the object on the test image set.
        results_object = model_object.predict(source=image_path, device=device_id, imgsz=meta_args['imgsz'], conf=0.25, iou=0.5)

        # Initialize the variable to hold the processed image.
        processed_image = image.copy()

        # If the model has found an object in the current processed image, express the results (class, bounding box, confidence).
        if results_object[0].boxes.shape[0] >= 1:
            # Express the data from the prediction:
            #   ID name of the class, Bounding box in the YOLO format and Confidence.
            class_id = results_object[0].boxes.cls.cpu().numpy(); b_box = results_object[0].boxes.xywhn.cpu().numpy()
            conf = results_object[0].boxes.conf.cpu().numpy()

            for _, (class_id_i, b_box_i, conf_i) in enumerate(zip(class_id, b_box, conf)):
                # Get the area of the rectangle.
                A = b_box_i[2] *  b_box_i[3]

                # If the calculated area of the object's bounding box is outside the limits, do not predict 
                # the object.
                if A < CONST_OBJECT_BB_AREA['Min'] or A > CONST_OBJECT_BB_AREA['Max']:
                    continue

                # If the confidence of the prediction is less than 90%, do not predict the object.
                if conf_i < 0.9:
                    continue

                # Create a bounding box from the label data.
                Bounding_Box_Properties = {'Name': f'{int(class_id_i)}', 'Precision': f'{str(conf_i)[0:5]}', 
                                           'Data': {'x_c': b_box_i[0], 'y_c': b_box_i[1], 'width': b_box_i[2], 'height': b_box_i[3]}}
                
                # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
                # the raw image.
                processed_image = Utilities.Image_Processing.Draw_Bounding_Box(image, Bounding_Box_Properties, 'YOLO', CONST_CONFIG_MODEL_OBJ['Color'][int(class_id_i)], 
                                                                               True, False)
                # Determine resolution of the processed image.
                img_h, img_w = image.shape[:2]
                Resolution = {'x': img_w, 'y': img_h}

                # Converts bounding box coordinates from YOLO format to absolute pixel coordinates.
                abs_coordinates_obj = Utilities.General.YOLO_To_Absolute_Coordinates({'x_c': b_box_i[0], 'y_c': b_box_i[1], 
                                                                                    'width': b_box_i[2], 'height': b_box_i[3]}, 
                                                                                    Resolution)
                    
                # Calculate object bounding box edges from center-based coordinates.
                obj_left = int(abs_coordinates_obj['x'] - abs_coordinates_obj['width'] / 2)
                obj_top = int(abs_coordinates_obj['y'] - abs_coordinates_obj['height'] / 2)
                obj_right = int(abs_coordinates_obj['x'] + abs_coordinates_obj['width'] / 2)
                obj_bottom = int(abs_coordinates_obj['y'] + abs_coordinates_obj['height'] / 2)

                # Crop the object region from the original image.
                cropped_image = image[obj_top:obj_bottom, obj_left:obj_right]

                # Perform defect detection only on specific object classes.
                #   Class ID (0) - Front side of the metalic object.
                if class_id_i == 0:
                    # Perform prediction of the defect on the test image set.
                    results_defect = model_defect.predict(source=cropped_image, device=device_id, imgsz=meta_args['imgsz'], conf=0.25, iou=0.5)

                    if results_defect[0].boxes.shape[0] >= 1:
                        # Express the data from the prediction of the defect.
                        defect_cls = results_defect[0].boxes.cls.cpu().numpy(); defect_b_box = results_defect[0].boxes.xywhn.cpu().numpy(); defect_conf = results_defect[0].boxes.conf.cpu().numpy()

                        for _, (d_class_i, d_b_box_i, d_conf_i) in enumerate(zip(defect_cls, defect_b_box, defect_conf)):
                            # If the confidence of the prediction is less than 80%, do not predict the object.
                            if d_conf_i < 0.8:
                                continue

                            # Convert bounding box of the defect to absolute coordinates within cropped object.
                            abs_coordinates_defect = Utilities.General.YOLO_To_Absolute_Coordinates({'x_c': d_b_box_i[0], 'y_c': d_b_box_i[1], 
                                                                                                     'width': d_b_box_i[2], 'height': d_b_box_i[3]}, 
                                                                                                    {'x': cropped_image.shape[1], 'y': cropped_image.shape[0]})

                            # Shift to original image.
                            abs_coordinates_defect['x'] += obj_left; abs_coordinates_defect['y'] += obj_top

                            # Generate YOLO-format label for original image.
                            yolo_coordinates_defect = Utilities.General.Absolute_Coordinates_To_YOLO(abs_coordinates_defect, Resolution)

                            # Create a bounding box from the label data of the defect.
                            Bounding_Box_Defect_Properties = {'Name': f'{int(d_class_i)}', 'Precision': f'{str(d_conf_i)[0:5]}', 
                                                              'Data': yolo_coordinates_defect}
                            
                            # Draw the bounding box of the defect with additional dependencies (name, precision, etc.) in 
                            # the raw image.
                            processed_image = Utilities.Image_Processing.Draw_Bounding_Box(processed_image, Bounding_Box_Defect_Properties, 'YOLO', CONST_CONFIG_MODEL_DEFECT['Color'][int(class_id_i)], 
                                                                                           True, False)
            
        # Save processed image.
        cv2.imwrite(f"{project_folder}/YOLO/Prediction/Dataset_v2_ALL/{image_name}.png", processed_image)

        # Release the image.
        del processed_image

        # Tracking the number of predictions.
        total_processed += 1

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print('[INFO] Data generation completed successfully.')
    print(f'[INFO] Total samples processed: {total_processed}')
    print(f'[INFO] Time: {int(minutes)}m {int(seconds)}s')

if __name__ == '__main__':
    sys.exit(main())