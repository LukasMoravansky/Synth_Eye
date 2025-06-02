import cv2
import numpy as np

# === Calibration Parameters ===
camera_matrix = np.array([
    [7163.39145, 0.0, 884.750393],
    [0.0, 7185.41497, 491.271378],
    [0.0, 0.0, 1.0]
])

dist_coeffs = np.array([
    -0.378557066, 28.1374230, -0.00651131808, -0.00121823634, 0.560603100
])

import os
project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'
img = cv2.imread(f'{project_folder}/Data/Camera/Basler/Image_{1:03}.png')

if img is None:
    print("Image not found. Check the path.")
    exit()

h, w = img.shape[:2]

# === Undistortion ===
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# === Crop the undistorted image to ROI ===
x, y, w, h = roi
undistorted_cropped = undistorted[y:y+h, x:x+w]

# === Save Undistorted Image ===
cv2.imwrite("undistorted.png", undistorted_cropped)
print("Undistorted image saved as 'undistorted.jpg'")
