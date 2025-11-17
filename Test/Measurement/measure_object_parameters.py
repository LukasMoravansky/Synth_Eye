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
#   ../Measurement/Core
import Measurement.Core

"""
Description:
    Initialization of constants.
"""
# The name of the dataset, model, and color of the object bounding boxes.
CONST_CONFIG = {'Name': 'Dataset_v2', 'Model': 'yolov8m_object_detection', 
                'Color': [(255, 165, 0), (0, 165, 255)]}

    # Object Dimensions (for measurement reference):
    # - Height: 60 mm
    # - Width: 40 mm
    # - Hole Diameter (Back side): 6 mm
    # - Hole Diameter (Front side): 12 mm
    # - Distance between holes (center-to-center): 25 mm

def main():
    """
    Description:
        ....
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

    # ...
    

    # Load a pre-trained custom YOLO model.
    model = YOLO(f"{project_folder}/YOLO/Model/{CONST_CONFIG['Name']}/{CONST_CONFIG['Model']}.pt")

    # Extracts numeric identifiers and background flags from filenames in a specified directory.
    image_dir_info = Utilities.General.Extract_Num_From_Filename('Image', os.path.join(project_folder, 'Data', CONST_CONFIG['Name'], 
                                                                                       'images', 'test'))

    # Initialize the timer and counters.
    start_time = time.time(); total_processed = 0

    for image_dir_info_i in image_dir_info:
        image_path = image_dir_info_i['Path']; image_name = image_dir_info_i['Name']

        # Load image.
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'[ERROR] Unable to load image from: {image_path}')

        # Perform prediction on the test image set.
        results = model.predict(source=image_path, device=device_id, imgsz=meta_args['imgsz'], conf=0.25, iou=0.5)

        # If the model has found an object in the current processed image, express the results (class, bounding box, confidence).
        if results[0].boxes.shape[0] >= 1:
            # Express the data from the prediction:
            #   ID name of the class, Bounding box in the YOLO format and Confidence.
            class_id = results[0].boxes.cls.cpu().numpy(); b_box = results[0].boxes.xywhn.cpu().numpy()
            conf = results[0].boxes.conf.cpu().numpy()

            for _, (class_id_i, b_box_i, conf_i) in enumerate(zip(class_id, b_box, conf)):
                Measurement_Obj_Param_Cls = Measurement.Core.Measurement_Object_Parameters_Cls(image=image,
                                                                                               object_bounding_box=b_box_i)

                processed_image = Measurement_Obj_Param_Cls.Cropped_Image.copy()
                image = processed_image.copy()

                # ...
                del Measurement_Obj_Param_Cls

        else:
            processed_image = image.copy()

        # Save processed image.
        cv2.imwrite(f"{project_folder}/Measurement/{CONST_CONFIG['Name']}/{image_name}.png", processed_image)

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