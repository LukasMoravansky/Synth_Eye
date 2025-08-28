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

"""
Description:
    Initialization of constants.
"""
# Configuration file ID to use for prediction.
CONST_CONFIGURATION_ID = 1

def main():
    """
    Description:
        Prediction (inference) using a trained YOLOv8 model on new images. The model is loaded from a checkpoint 
        and applied to a test set to detect and classify objects.

        For more information, see:
            https://docs.ultralytics.com/modes/predict/
    """

    # Locate the path to the project root.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Determine computation device.
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

    # Load trained YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Results/{dataset_name}/train_fb_{CONST_FREEZE_BACKBONE}/weights/best.pt')

    # Perform prediction on the test image set.
    _ = model.predict(
        source=os.path.join(meta_args[1]['path'], 'images', 'test'),
        imgsz=meta_args[0]['imgsz'],
        conf=0.25,
        iou=0.5,
        device=device_id,
        save=True,
        save_txt=True,
        save_conf=True,
        name=f'{project_folder}/YOLO/Results/{dataset_name}/predict_fb_{CONST_FREEZE_BACKBONE}'
    )

if __name__ == '__main__':
    main()
