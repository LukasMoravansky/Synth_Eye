# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# OS (Operating system interfaces)
import os

import torch

"""
Description:
    Initialization of constants.
"""

# An indication of whether the backbone layers of the model should be frozen.
CONST_FREEZE_BACKBONE = False

def main():
    """
    Description:
        Validation of the YOLOv8 model after training. In this case, the model is evaluated on a test dataset to measure 
        its accuracy and generalization performance.

        For more information see: 
            https://docs.ultralytics.com/modes/val/
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Results/Dataset_v2/train_fb_{CONST_FREEZE_BACKBONE}/weights/best.pt')

    # Evaluate the performance of the model on the validation dataset.
    _ = model.val(
        data=f'{project_folder}/YOLO/Configuration/Cfg_Model_1.yaml',
        imgsz=640,
        batch=16,
        device=0,
        verbose=True,
        name=f'{project_folder}/YOLO/Results/Dataset_v2/valid_fb_{CONST_FREEZE_BACKBONE}'
    )

if __name__ == '__main__':
    main()