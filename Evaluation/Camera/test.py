# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OpenCV library for computer vision tasks
import cv2
# OS module for file handling and accessing directories
import os
# Library to work with Basler cameras
from Basler.Camera import Basler_Cls

def main():
    # Locate the path to the project folder
    project_folder = os.getcwd().split('BIW_Vision_AI')[0] + 'BIW_Vision_AI'

    # Custom camera configuration.
    custom_cfg = {
        'exposure_time': 1000,
        'gain': 10,
        'balance_ratios': {'Red': 1.1, 'Green': 1.0, 'Blue': 1.3},
        'pixel_format': 'BayerRG8'
    }

    # Initialize and configure the Basler camera.
    Basler_Cam_Id_1 = Basler_Cls(config=custom_cfg)

    while True:
        # Capture a single image.
        img_raw = Basler_Cam_Id_1.Capture()

        if img_raw is None:
            print("No image captured!")
            break
        
        # Convert the image from BGR to RGB color format
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        img_height, img_width, _ = img_rgb.shape

        # Show the captured image.
        cv2.imshow("Captured Image", img_raw)
        
        # Wait for the user to press the 'c' key to exit the loop or stop if the camera is no longer grabbing.
        key = cv2.waitKey(100) & 0xFF
        if key == ord('c') or Basler_Cam_Id_1.Is_Grabbing == True:
            print("Exiting the capture loop.")
            break

    # Release the camera resources and close the OpenCV window.
    cv2.destroyAllWindows()
    del Basler_Cam_Id_1

if __name__ == '__main__':
    sys.exit(main())