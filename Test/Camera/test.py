import cv2
import numpy as np
import sys
sys.path.append('..')
import Dataset.Utils as Utils

# https://calib.io/pages/camera-calibration-pattern-generator

# === Configuration ===
CHECKERBOARD = (11-1,8-1)   # (columns, rows) of inner corners
SQUARE_SIZE = 12.0       # square size in mm

# === Prepare 3D object points ===
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D world points
imgpoints = []  # 2D image points

# === Load image ===
"""
Pixel to mm conversion:
  → Horizontal: 1 px = 0.09744 mm (avg 123.16 px per 10 mm)
  → Vertical:   1 px = 0.09730 mm (avg 123.33 px per 10 mm)
"""

img_n = 'RS'
#image = cv2.imread("C:\projects\Synth_Eye\Data\Camera\Basler\Image_099.png")
image = cv2.imread(r'C:\projects\Synth_Eye\Data\Camera\Basler_a2A1920_51gcPRO_Computar_M1228_MPW3_Virtual\Checkerboard.png')
image = Utils.process_synthetic_image(image.copy())
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === Find corners ===
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
if ret:
    # Refine corner positions
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    objpoints.append(objp)
    imgpoints.append(corners_subpix)

    # === Draw and save the detected checkerboard ===
    vis = cv2.drawChessboardCorners(image.copy(), CHECKERBOARD, corners_subpix, ret)
    cv2.imwrite(f"Detected_Checkerboard_{img_n}.png", vis)

    # === Camera Calibration ===
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Calibration successful!")
    print("Camera Matrix:\n", cameraMatrix)
    print("Distortion Coefficients:\n", distCoeffs.ravel())

    # === Compute pixel-to-mm ratio in X and Y directions ===
    def compute_mm_per_pixel(corners, cols, rows, square_size):
        horizontal_dists = []
        vertical_dists = []

        for row in range(rows):
            for col in range(cols - 1):
                i = row * cols + col
                pt1 = corners[i][0]
                pt2 = corners[i + 1][0]
                dist = np.linalg.norm(pt2 - pt1)
                horizontal_dists.append(dist)

        for col in range(cols):
            for row in range(rows - 1):
                i = row * cols + col
                pt1 = corners[i][0]
                pt2 = corners[i + cols][0]
                dist = np.linalg.norm(pt2 - pt1)
                vertical_dists.append(dist)

        avg_dx = np.mean(horizontal_dists)
        avg_dy = np.mean(vertical_dists)
        mm_per_px_x = square_size / avg_dx
        mm_per_px_y = square_size / avg_dy

        return mm_per_px_x, mm_per_px_y, avg_dx, avg_dy

    mm_per_px_x, mm_per_px_y, avg_px_x, avg_px_y = compute_mm_per_pixel(
        corners_subpix, CHECKERBOARD[0], CHECKERBOARD[1], SQUARE_SIZE
    )

    print(f"Pixel to mm conversion:")
    print(f"  → Horizontal: 1 px = {mm_per_px_x:.5f} mm (avg {avg_px_x:.2f} px per 10 mm)")
    print(f"  → Vertical:   1 px = {mm_per_px_y:.5f} mm (avg {avg_px_y:.2f} px per 10 mm)")

else:
    print("Checkerboard not detected.")