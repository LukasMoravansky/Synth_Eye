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

    # Generate a random position of the object.
    Object_Cls.Random()
    
    # Initialize the material class for randomizing material properties and handling 
    # baking operations.
    Material_Cls = Blender.Core.Material_Cls('BakeTemp128')

    # Generate random material properties and process bounding box detection.
    #   Note:
    #       If the returned information is None, it indicates that the bounding box 
    #       has not been generated for the material.
    material_info_b_box = Material_Cls.Random('Area_Testing_Mat')
    
    # Generate random camera properties.
    Camera_Cls.Random()
    
    # Get the 2D coordinates of the bounding box from the rendered object scanned by the camera.
    bounding_box_2d = Utilities.General.Get_2D_Coordinates_Bounding_Box(Object_Cls.Vertices, Camera_Cls.P(), 
                                                                        Parameters.Scene.Basler_Cam_Str.Resolution, 'YOLO')
    
    # Check if the object's rotation along the Y-axis indicates it is flipped.
    if bpy.data.objects[Object_Cls.Name].rotation_euler.z < Mathematics.CONST_MATH_HALF_PI:
        if material_info_b_box == None:
            cls_id = np.array([0],dtype=int); b_box_2d = np.array([list(bounding_box_2d.values())])
        else:
            cls_id = np.array([0,2],dtype=int); b_box_2d = np.array([list(bounding_box_2d.values()),
                                                                     material_info_b_box])
            Blender.Utilities.Save_Synthetic_Data(f'{project_folder}/Data/Dataset_v1/', 'train', f'{CONST_INIT_INDEX:03}', 
                                                  cls_id, b_box_2d, 'txt', 'png')
    else:
        cls_id = np.array([1],dtype=int); b_box_2d = np.array([list(bounding_box_2d.values())])

    print(cls_id)
    print(b_box_2d)
    """ 
    # Save the image with the corresponding label.
    Blender.Utilities.Save_Synthetic_Data(f'{project_folder}/Data/Dataset_v1/', 'train', f'{CONST_INIT_INDEX:03}', 
                                          cls_id, b_box_2d, 'txt', 'png')                                    
    """
    
    # Release the classes.
    del Camera_Cls, Object_Cls, Material_Cls

if __name__ == '__main__':
    main()
