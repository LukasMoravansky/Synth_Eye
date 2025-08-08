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
# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 1
# The name of the dataset and color of the object bounding boxes.
#   Dataset_v2 - [(255, 165, 0), (0, 165, 255)]
#   Dataset_v3 - [(80, 0, 255)]
CONST_DATASET = {'Name': 'Dataset_v2', 'Color': [(255, 165, 0), (0, 165, 255)]}

def main():
    """
    Description:
        A program to perform AI-based object detection on a test image dataset using a pre-trained 
        YOLOv8 model.

        It loads test images, runs inference, draws bounding boxes around detected 
        objects, and saves the annotated images.

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
    model = YOLO(f'{project_folder}/YOLO/Model/{CONST_DATASET['Name']}/yolov8m_object_detection.pt')

    # Extracts numeric identifiers and background flags from filenames in a specified directory.
    image_dir_info = Utilities.General.Extract_Num_From_Filename('Image', os.path.join(project_folder, 'Data', CONST_DATASET['Name'], 
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
                # Create a bounding box from the label data.
                Bounding_Box_Properties = {'Name': f'{int(class_id_i)}', 'Precision': f'{str(conf_i)[0:5]}', 
                                           'Data': {'x_c': b_box_i[0], 'y_c': b_box_i[1], 'width': b_box_i[2], 'height': b_box_i[3]}}
                
                # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
                # the raw image.
                processed_image = Utilities.Image_Processing.Draw_Bounding_Box(image, Bounding_Box_Properties, 'YOLO', CONST_DATASET['Color'][int(class_id_i)], 
                                                                               True, True)
        else:
            processed_image = image.copy()
            
        # Save processed image.
        cv2.imwrite(f'{project_folder}/YOLO/Prediction/{CONST_DATASET['Name']}/{image_name}.png', processed_image)

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
    main()