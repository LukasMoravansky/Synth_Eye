# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# OS (Operating system interfaces)
import os

"""
Description:
    Initialization of constants.
"""
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

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Results/Dataset_v3/train_fb_{CONST_FREEZE_BACKBONE}/weights/best.pt')

    # Predict (test) the model on a test dataset.
    _ = model.predict(
        source=f'{project_folder}/Data/Dataset_v3/images/test',
        imgsz=640,
        conf=0.25,
        iou=0.5,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        name=f'{project_folder}/YOLO/Results/Dataset_v3/predict_fb_{CONST_FREEZE_BACKBONE}'
    )

if __name__ == '__main__':
    main()