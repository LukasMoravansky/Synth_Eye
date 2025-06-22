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
        Prediction (testing) using the trained YOLOv8 model on new images. In this case, the model is loaded from a checkpoint 
        file and the user can provide images to perform inference. The model predicts the classes and locations of objects 
        in the input images.

        For more information see: 
            https://docs.ultralytics.com/modes/predict/
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Automatically select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Results/Dataset_v1/train_fb_{CONST_FREEZE_BACKBONE}/weights/best.pt')

    # Predict (test) the model on a test dataset.
    _ = model.predict(
        source=f'{project_folder}/Data/Dataset_v1/images/test',
        imgsz=1280,
        conf=0.25,                  # Higher confidence threshold for deployment
        iou=0.45,                   # Standard NMS IoU threshold for inference
        device=device,
        save=True,                  # Save predicted images with bounding boxes
        save_txt=True,              # Save YOLO format predicted labels
        save_conf=True,             # Include confidence scores in saved labels
        name=f'{project_folder}/YOLO/Results/Dataset_v1/predict_fb_{CONST_FREEZE_BACKBONE}'
    )

if __name__ == '__main__':
    main()