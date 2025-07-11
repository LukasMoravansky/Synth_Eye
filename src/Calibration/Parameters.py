# Numpy (Array computing)
import numpy as np
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Typing (Support for type hints)
import typing as tp

@dataclass
class Camera_Calibration_Parameters_Str:
    """
    Description:
        The structure of the main parameters of the camera calibration
    """

    # Camera calibration matrix.
    K: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    # Distortion coefficients: [k1, k2, p1, p2, k3]
    Coefficients: tp.Dict[str, float] = field(default_factory=dict)
    # Pixel to millimeter conversion factors for x and y axes.
    Conversion_Factor: tp.Dict[str, float] = field(default_factory=dict)

"""
Description:
    Calibration parameters of the Basler a2A1920-51gcPRO camera using 
    the Computar M1228-MPW3 lens.
"""
Basler_Calib_Param_Str = Camera_Calibration_Parameters_Str()
Basler_Calib_Param_Str.K = np.array([
    [7163.39148, 0.0, 884.750383],
    [0.0, 7185.41499, 491.271296],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
Basler_Calib_Param_Str.Coefficients = {
    'k1': -0.378556984,
    'k2': 28.1374127,
    'p1': -0.00651131765,
    'p2': -0.00121823652,
    'k3': 0.560603312
}
Basler_Calib_Param_Str.Conversion_Factor = {
    'x': 0.09744,
    'y': 0.09730
}