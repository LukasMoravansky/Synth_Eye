# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# OS (Operating system interfaces)
import os
# Custom Library:
#   ../Utilities/Model
import Utilities.Model

import torch

"""
Description:
    Initialization of constants.
"""
# Select the desired size of YOLOv* to build the model.
#   Note:
#     Detection Model.
#   Nano: 'yolov8n', Small: 'yolov8s', Medium: 'yolov8m', Large: 'yolov8l', XLarge: 'yolov8x'}
CONST_YOLO_SIZE = 'yolov8m'
# An indication of whether the backbone layers of the model should be frozen.
CONST_FREEZE_BACKBONE = False

def main():
    """
    Description:
        Training the YOLOv8 model on a custom dataset. In this case, the model is trained using the specified dataset 
        and hyperparameters. 

        For more information see: 
            https://docs.ultralytics.com/modes/train/#arguments

        Warning:
            The config.yaml file needs to be changed to allow access to the path (internal/google colab) to the dataset 
            to be used for training.
                ../YOLO/Configuration/Type_{dataset_type}/config.yaml
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Remove the YOLO model, if it already exists.
    if os.path.isfile(f'{CONST_YOLO_SIZE}.pt'):
        print(f'[INFO] Removing the YOLO model.')
        os.remove(f'{CONST_YOLO_SIZE}.pt')

    # Load a pre-trained YOLO model.
    model = YOLO(f'{CONST_YOLO_SIZE}.pt')

    if CONST_FREEZE_BACKBONE == True:
        # Triggered when the training starts.
        model.add_callback('on_train_start', Utilities.Model.Freeze_Backbone)

    # Training the model on a custom dataset with additional dependencies (number of epochs, image size, etc.)
    model.train(
        data=f'{project_folder}/YOLO/Configuration/Cfg_Model_1.yaml',
        batch=16,
        imgsz=640,
        device=0,
        epochs=300,
        patience=0,
        rect=True,
        name=f'{project_folder}/YOLO/Results/Dataset_v2/train_fb_{CONST_FREEZE_BACKBONE}',
        lr0=0.001,
        warmup_epochs=5,
        cos_lr=True,
        mosaic=0.2,
        close_mosaic=50,
        hsv_h=0.015,
        hsv_s=0.2,
        hsv_v=0.2,
        fliplr=0.5,
        flipud=0.0,
        scale=0.05,
        translate=0.05,
        amp=True,
        cache='disk',
        workers=8,
        freeze=0,
        val=True,
        verbose=True
    )

    # augment=True ???

if __name__ == '__main__':
    main()  