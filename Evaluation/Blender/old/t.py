# BPY (Blender as a python)
import bpy
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# OS (Operating system interfaces)
import os
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
#   ../Transformation/Utilities
import Transformation.Utilities.Mathematics as Mathematics

# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 1

def main():
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'
    
    # Deselect all objects in the current scene.
    Blender.Utilities.Deselect_All()
    
    # Initialize the camera with lighting to capture an object in the scene.
    #   The main parameters of the camera and light can be found at: ../Parameters/Scene.py
    Camera_Cls = Blender.Core.Camera_Cls(Parameters.Scene.Basler_Cam_Str, Parameters.Scene.Effilux_Light_Str, 'PNG')

    # Initialize the object to be captured by the camera.
    #   The main parameters of the object can be found at: ../Parameters/Object.py
    Object_Cls = Blender.Core.Object_Cls(Parameters.Object.Object_001_Str, 'ZYX')
    
    # Turn on/off visibility of the object.
    Object_Cls.Visibility(True)
    
    # Initialize the material class for randomizing material properties and handling 
    # baking operations.
    Material_Cls = Blender.Core.Material_Cls('UV_Scaled_128')

    # Generate a random position of the object.
    Object_Cls.Random()
    #   ....
    P_extended = np.vstack((Camera_Cls.P(), np.ones(4)))
    obj_center_px_tmp = (P_extended @ np.hstack((np.array(Object_Cls.T.p.all()), 1)))[0:-1]
    obj_center_px = np.array(obj_center_px_tmp/obj_center_px_tmp[-1], dtype=int)[0:-1]

    # Generate random camera properties.
    Camera_Cls.Random()

    # ...
    obj_is_flipped = not np.abs(Object_Cls.T.Get_Rotation('ZYX')[2]) < Mathematics.CONST_MATH_HALF_PI

    # Generate random material properties and process bounding box detection.
    #   Note:
    #       If the returned information is None, it indicates that the bounding box 
    #       has not been generated for the material.
    material_info_b_box = Material_Cls.Random('Area_Testing_Mat', list(Parameters.Scene.Basler_Cam_Str.Resolution.values()),
                                              obj_is_flipped, obj_center_px, Object_Cls.T.Get_Rotation('ZYX')[2])
    
    # Get the 2D coordinates of the bounding box from the rendered object scanned by the camera.
    bounding_box_2d = Utilities.General.Get_2D_Coordinates_Bounding_Box(Object_Cls.Vertices, Camera_Cls.P(), 
                                                                        Parameters.Scene.Basler_Cam_Str.Resolution, 'YOLO')
    
    # Check if the object's rotation along the Y-axis indicates it is flipped.
    if np.abs(Object_Cls.T.Get_Rotation('ZYX')[2]) < Mathematics.CONST_MATH_HALF_PI:
        if material_info_b_box == None:
            cls_id = np.array([0],dtype=int); b_box_2d = np.array([list(bounding_box_2d.values())])
        else:
            bb_fingerprint = Utilities.General.Convert_Boundig_Box_Data('PASCAL_VOC', 'YOLO', {'x_min': material_info_b_box[0], 'y_min': material_info_b_box[2], 
                                                                                               'x_max': material_info_b_box[1], 'y_max': material_info_b_box[3]}, 
                                                                                             Parameters.Scene.Basler_Cam_Str.Resolution)
            cls_id = np.array([0,2],dtype=int); b_box_2d = np.array([list(bounding_box_2d.values()),
                                                                     list(bb_fingerprint.values())])
    else:
        cls_id = np.array([1],dtype=int); b_box_2d = np.array([list(bounding_box_2d.values())])


    """ 
    # Save the image with the corresponding label.
    Blender.Utilities.Save_Synthetic_Data(f'{project_folder}/Data/Dataset_v1/', 'train', f'{CONST_INIT_INDEX:03}', 
                                          cls_id, b_box_2d, 'txt', 'png')                                    
    """
    
    # Release the classes.
    del Camera_Cls, Object_Cls, Material_Cls

if __name__ == '__main__':
    main()
