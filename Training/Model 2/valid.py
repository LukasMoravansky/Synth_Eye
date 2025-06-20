# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# OS (Operating system interfaces)
import os

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
CONST_FREEZE_BACKBONE = True

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
    model = YOLO(f'{project_folder}/YOLO/Results/Dataset_v1/train_fb_{CONST_FREEZE_BACKBONE}/weights/best.pt')

    # Evaluate the performance of the model on the validation dataset.
    model.val(
        data=f'{project_folder}/YOLO/Configuration/Cfg_Model_2.yaml',
        batch=16,
        imgsz=640,
        device='cuda',
        conf=0.1,                   # Lower conf to catch subtle defects
        iou=0.45,                   # Standard IoU for NMS in defect detection
        save_txt=True,
        save_conf=True,
        save_json=False,
        split='val',
        name=f'{project_folder}/YOLO/Results/Dataset_v2/valid_fb_{CONST_FREEZE_BACKBONE}'
    )

if __name__ == '__main__':
    main()