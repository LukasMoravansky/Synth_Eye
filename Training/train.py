# System (Default)
import sys
# Add ../src to system path if not already present.
if '../src' not in sys.path:
    sys.path.append('../src')
# Ultralytics (real-time object detection and image segmentation)
# [pip install ultralytics]
from ultralytics import YOLO
# OS (operating system interfaces)
import os
# YAML (configuration file parsing)
import yaml
# PyTorch (tensors and dynamic neural networks)
import torch
# Custom Library:
#   ../Utilities/Model
import Utilities.Model

"""
Description:
    Initialization of constants.
"""
# Select the desired size of YOLOv* to build the model.
#   Nano: 'yolov8n', Small: 'yolov8s', Medium: 'yolov8m', Large: 'yolov8l', XLarge: 'yolov8x'}
CONST_YOLO_SIZE = 'yolov8m'
# The identification number of the configuration file for training the NN model.
CONST_CONFIGURATION_ID = 1

def main():
    """
    Description:
        Training the YOLOv8 model on a custom dataset. The model is trained using the specified dataset 
        and hyperparameters loaded from YAML configs.

        For more information see: 
            https://docs.ultralytics.com/modes/train/#arguments

        Warning:
            The dataset path inside Cfg_Model_{id}.yaml must be updated to match 
            your local or Colab environment.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Remove the pre-existing YOLO model file if it exists
    if os.path.isfile(f'{CONST_YOLO_SIZE}.pt'):
        print(f'[INFO] Removing the YOLO model.')
        os.remove(f'{CONST_YOLO_SIZE}.pt')

    # Detect and assign the training device (GPU if available).
    device_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device_id}")
    if device_id.type == 'cuda':
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

    # Load hyperparameters and dataset configuration files.
    meta_args = []
    for cfg_name_i in ['Args_Model', f'{project_folder}/YOLO/Configuration/Cfg_Model']:
        with open(f'{cfg_name_i}_{CONST_CONFIGURATION_ID}.yaml', 'r') as f:
            meta_args.append(yaml.safe_load(f))

    # Extract dataset name (e.g., "Dataset_v2") from config path.
    dataset_name = os.path.basename(meta_args[1]['path'])

    # An indication of whether the backbone layers of the model should be frozen.
    if meta_args[0]['freeze'] > 0:
        CONST_FREEZE_BACKBONE = True
    else:
        CONST_FREEZE_BACKBONE = False
        
    # Update the dataset config path and training output directory.
    meta_args[0]['data'] = f'{project_folder}/YOLO/Configuration/Cfg_Model_{CONST_CONFIGURATION_ID}.yaml'
    meta_args[0]['name'] = f'{project_folder}/YOLO/Results/{dataset_name}/train_fb_{CONST_FREEZE_BACKBONE}'
    meta_args[0]['device'] = device_id

    # Load a pre-trained YOLO model.
    model = YOLO(f'{CONST_YOLO_SIZE}.pt')

    # Optionally: Freeze the model backbone during training.
    if CONST_FREEZE_BACKBONE == True:
        model.add_callback('on_train_start', Utilities.Model.Freeze_Backbone)

    # Training the model on a custom dataset with additional dependencies.
    #   Note:
    #       Additional hyperparameters are loaded from Args_Model_{id}.yaml.
    model.train(**meta_args[0])

if __name__ == '__main__':
    main()  