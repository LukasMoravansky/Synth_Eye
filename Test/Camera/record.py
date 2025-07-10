# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OpenCV library for computer vision tasks
import cv2
# Custom Lib.:
#   ../Basler/Camera
from Basler.Camera import Basler_Cls
#   ../Calibration/Parameters
from Calibration.Parameters import Basler_Calib_Param_Str

def main():
    """
    Description:
        A program to configure a Basler camera (a2A1920-51gcPRO) with custom settings for continuous image capture. The system is equipped
        with the EFFI-FD-200-200-000 lighting for optimal illumination in the vision stand.

        Setup:
            Camera Model: Basler a2A1920-51gcPRO GigE Camera
            Lighting: EFFI-FD-200-200-000 High-Power Flat Light
    """
        
    # Custom camera configuration.
    custom_cfg = {
        'exposure_time': 10000,
        'gain': 10,
        'balance_ratios': {'Red': 0.95, 'Green': 0.9, 'Blue': 1.2},
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
        
        # Undistort the image using camera calibration parameters.
        h, w = img_raw.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(Basler_Calib_Param_Str.K, Basler_Calib_Param_Str.Coefficients, (w, h), 1, (w, h))
        img_undistorted = cv2.undistort(img_raw, Basler_Calib_Param_Str.K, Basler_Calib_Param_Str.Coefficients, None, new_camera_matrix)

        # Show the captured image.
        cv2.imshow("Captured Image", img_undistorted)
        
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