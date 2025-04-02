# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OS (Operating system interfaces)
import os
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# Custom Library:
#   ../Utilities/Image_Processing
import Utilities.Image_Processing
#   ../Utilities/File_IO
import Utilities.File_IO as File_IO

"""
Description:
    Initialization of constants.
"""
# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 301
# The color of the bounding box of the object.
CONST_OBJECT_BB_COLOR = [(255, 165, 0), (0, 165, 255), (80, 0, 255)]

CONST_OBJECT_NAME = ['Cls_Front_Side', 'Cls_Back_Side', 'Cls_Fingerprint']

def main():
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Results/Dataset_v1/train_fb_True/weights/best.pt')

    # Load a raw image from a file.
    image_data = cv2.imread(f'{project_folder}/Data/Dataset_v1/images/test/Image_{CONST_INIT_INDEX:03}.png')

    # Predict (test) the model on a test dataset.
    results = model.predict(source=f'{project_folder}/Data/Dataset_v1/images/test/Image_{CONST_INIT_INDEX:03}.png', imgsz=640, conf=0.5, iou=0.7)

    # If the model has found an object in the current processed image, express the results (class, bounding box, confidence).
    if results[0].boxes.shape[0] >= 1:
        # Express the data from the prediction:
        #   ID name of the class, Bounding box in the YOLO format and Confidence.
        class_id = results[0].boxes.cls.cpu().numpy(); b_box = results[0].boxes.xywhn.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()

        for i, (class_id_i, b_box_i, conf_i) in enumerate(zip(class_id, b_box, conf)):
            if class_id_i == 2 and conf_i < 0.9:
                # Create a bounding box from the label data.
                Bounding_Box_Properties = {'Name': f'{CONST_OBJECT_NAME[int(class_id_i)]}_{i}', 'Precision': f'{str(conf_i)[0:5]}', 
                                           'Data': {'x_c': b_box_i[0], 'y_c': b_box_i[1], 'width': b_box_i[2], 'height': b_box_i[3]}}
                
                # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
                # the raw image.
                image_data = Utilities.Image_Processing.Draw_Bounding_Box(image_data, Bounding_Box_Properties, 'YOLO', CONST_OBJECT_BB_COLOR[int(class_id_i)], 
                                                                          True, True)
            elif class_id_i == 0:
                # Create a bounding box from the label data.
                Bounding_Box_Properties = {'Name': f'{CONST_OBJECT_NAME[int(class_id_i)]}', 'Precision': f'{str(conf_i)[0:5]}', 
                                           'Data': {'x_c': b_box_i[0], 'y_c': b_box_i[1], 'width': b_box_i[2], 'height': b_box_i[3]}}
                
                # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
                # the raw image.
                image_data = Utilities.Image_Processing.Draw_Bounding_Box(image_data, Bounding_Box_Properties, 'YOLO', CONST_OBJECT_BB_COLOR[int(class_id_i)], 
                                                                          True, True)

                    
        # Saves the images to the specified file.
        cv2.imwrite(f'Image_Res_{CONST_INIT_INDEX:03}.png', image_data)
            
if __name__ == '__main__':
    main()