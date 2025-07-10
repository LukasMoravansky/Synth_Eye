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
# Whether to freeze the model's backbone during training.
CONST_FREEZE_BACKBONE = False
# Configuration file ID to load dataset and model parameters.
CONST_CONFIGURATION_ID = 1

def main():
    """
    Description:
        Validates a trained YOLOv8 model on a dataset to evaluate its accuracy and generalization performance.

        For more information, see:
            https://docs.ultralytics.com/modes/val/
    """

    # Locate the path to the project root.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Set device based on CUDA availability.
    device_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device_id}")
    if device_id.type == 'cuda':
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

    # Load configuration metadata.
    with open(f'{project_folder}/YOLO/Configuration/Cfg_Model_{CONST_CONFIGURATION_ID}.yaml', 'r') as f:
        meta_args = yaml.safe_load(f)

    # Extract dataset name from the config path.
    dataset_name = os.path.basename(meta_args['path'])

    # Load the best model weights from training.
    model = YOLO(f'{project_folder}/YOLO/Results/{dataset_name}/train_fb_{CONST_FREEZE_BACKBONE}/weights/best.pt')

    # Run model validation using the specified dataset and parameters.
    _ = model.val(
        data=f'{project_folder}/YOLO/Configuration/Cfg_Model_{CONST_CONFIGURATION_ID}.yaml',
        imgsz=640,
        batch=16,
        device=device_id,
        verbose=True,
        name=f'{project_folder}/YOLO/Results/{dataset_name}/valid_fb_{CONST_FREEZE_BACKBONE}'
    )

if __name__ == '__main__':
    main()