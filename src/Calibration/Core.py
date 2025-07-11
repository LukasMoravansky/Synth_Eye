# OpenCV library for computer vision tasks
import cv2
# Numpy (Array computing)
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Calibration/Parameters
from Calibration.Parameters import Camera_Calibration_Parameters_Str

class Checkerboard_Calibration_Cls:
    def __init__(self, inner_corners: tp.Tuple[int, int], square_size: float):
        self.__inner_corners = inner_corners
        self.__square_size = square_size
        self.__Calib_Param_Str = Camera_Calibration_Parameters_Str()

        self.__obj_points = np.zeros((self.__inner_corners[0] * self.__inner_corners[1], 3), 
                                     np.float32)
        self.__obj_points[:, :2] = np.mgrid[0:self.__inner_corners[0],
                                            0:self.__inner_corners[1]].T.reshape(-1, 2)
        self.__obj_points *= self.__square_size

    def __Get_Pixel_To_MM_Factor(self, corner_subpix: np.ndarray) -> tp.Dict[str, float]:
        """
        Description:
            Computes pixel to millimeter conversion factors in the X and Y directions
            based on refined subpixel corner detections.

        Args:
            (1) corners_subpix [np.ndarray]: Refined corner positions, shape (N, 1, 2).

        Returns:
            (1) parameter Dict[str, float]: Conversion factor in horizontal and vertical direction.
        """

        dist_horizontal = []; dist_vertical = []

        # Measure horizontal distances between adjacent columns in each row.
        for row_i in range(self.__inner_corners[0]):
            for column_i in range(self.__inner_corners[1] - 1):
                idx = row_i * self.__inner_corners[1] + column_i
                dist_horizontal.append(np.linalg.norm(corner_subpix[idx + 1][0] 
                                                      - corner_subpix[idx][0]))

        # Measure vertical distances between adjacent rows in each column
        for column_i in range(self.__inner_corners[1]):
            for row_i in range(self.__inner_corners[0] - 1):
                idx = row_i * self.__inner_corners[1] + column_i
                dist_vertical.append(np.linalg.norm(corner_subpix[idx + self.__inner_corners[1]][0] 
                                                    - corner_subpix[idx][0]))

        return {
            'x': self.__square_size / float(np.mean(dist_horizontal)),
            'y': self.__square_size / float(np.mean(dist_vertical))
        }

    def Solve(self, image: np.ndarray, draw_chessboard_corners: bool = False, path: str = '') -> tp.Tuple[bool, tp.Optional[Camera_Calibration_Parameters_Str]]:
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image.copy()

        ret, corners = cv2.findChessboardCorners(img_gray, self.__inner_corners, None)
        if not ret:
            print('[INFO] Checkerboard not detected.')
            return False, None

        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

        _, K, dist_coeffs_tmp, _, _ = cv2.calibrateCamera(
            [self.__obj_points], [corners_subpix], img_gray.shape[::-1], None, None
        )

        dist_coeffs = dist_coeffs_tmp.ravel()

        self.__Calib_Param_Str.K = K
        self.__Calib_Param_Str.Coefficients = {
            'k1': dist_coeffs[0],
            'k2': dist_coeffs[1],
            'p1': dist_coeffs[2],
            'p2': dist_coeffs[3],
            'k3': dist_coeffs[4] if dist_coeffs.size > 4 else 0.0
        }

        # Determine the pixel to mm conversion factor in both directions from the actual corners.
        self.__Calib_Param_Str.Conversion_Factor = self.__Get_Pixel_To_MM_Factor(corners_subpix)

        if draw_chessboard_corners == True:
            img_corners = cv2.drawChessboardCorners(img_gray, self.__inner_corners, corners_subpix, ret)
            cv2.imwrite(f'{path}/Image_Checkerboard_Detected.png', img_corners)

        return True, self.__Calib_Param_Str
