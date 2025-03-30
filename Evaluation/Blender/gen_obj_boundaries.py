# BPY (Blender as a python)
import bpy
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# Numpy (Array computing)
import numpy as np
# Custom Lib.:
#   ../Blender/Core & Utilities
import Blender.Core
import Blender.Utilities
#   ../Parameters/Scene & Object
import Parameters.Scene
import Parameters.Object
#   ../Utilities/General
import Utilities.General

"""
Description:
    Open *.blend file from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd ../{project_name}/Blender
        $ blender {blender_file_name}.blend
"""

"""
Description:
    Initialization of constants.
"""
# Number of data to be generated.
#   Note:
#       More data could be useful for better accuracy.
CONST_NUM_OF_DATA = 10000

def main():
    """
    Description:
        A program to get the boundaries of an object (bounding box). More precisely the area of the rectangle.

        Rectangle Area:
            A = w * h,

            where w is the width and h is the height of the 2D coordinates of the bounding box.

        Boundaries (limits):
            A_{-}, A_{+}
    """

    # Deselect all objects in the current scene.
    Blender.Utilities.Deselect_All()

    # Initialize the camera with lighting to capture an object in the scene.
    #   The main parameters of the camera and light can be found at: ../Parameters/Scene.py
    Camera_Cls = Blender.Core.Camera_Cls(Parameters.Scene.Basler_Cam_Str, Parameters.Scene.Effilux_Light_Str, 'PNG')

    # Initialize the object to be captured by the camera.
    #   The main parameters of the object can be found at: ../Parameters/Object.py
    Object_Cls = Blender.Core.Object_Cls(Parameters.Object.Object_001_Str, 'ZYX')
    
    # Enable (turn on) visibility of the object.
    Object_Cls.Visibility(True)

    # Generates data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_DATA}.
    i = 0; A = []
    while CONST_NUM_OF_DATA > i:
        # Generate a random position of the object.
        Object_Cls.Random()

        # Get the 2D coordinates of the bounding box from the rendered object scanned by the camera.
        bounding_box_2d = Utilities.General.Get_2D_Coordinates_Bounding_Box(Object_Cls.Vertices, Camera_Cls.P(), 
                                                                            Parameters.Scene.Basler_Cam_Str.Resolution, 'YOLO')
        
        # Get the area of the rectangle.
        A.append(bounding_box_2d['width'] * bounding_box_2d['height'])
        i += 1
        
    # Display information.
    print('[INFO] The average area of a rectangle:')
    print(f'[INFO]   A = {np.sum(A)/len(A)}')
    print('[INFO] Boundaries (limits):')
    print(f'[INFO]   A_[-] = {np.min(A)}')
    print(f'[INFO]   A_[+] = {np.max(A)}')

    # Disable (turn off) visibility of the object.
    Object_Cls.Visibility(False)

if __name__ == '__main__':
    main()