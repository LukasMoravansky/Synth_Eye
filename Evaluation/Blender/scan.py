# BPY (Blender as a python)
import bpy
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# OS (Operating system interfaces)
import os
# Custom Lib.:
#   ../Blender/Core & Utilities
import Blender.Core
import Blender.Utilities
#   ../Parameters/Scene & Object
import Parameters.Scene
import Parameters.Object

"""
Description:
    Open *.blend file from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd ../{project_name}/Blender
        $ blender {blender_file_name}.blend
"""

# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 0

def main():
    """
    Description:
        A program to configure a virtual Basler camera (a2A1920-51gcPRO) with custom settings to capture an image. The system is equipped 
        with the virtual EFFI-FD-200-200-000 lighting for optimal illumination in the virtual vision stand.

        Virtual Setup:
            Camera Model: Basler a2A1920-51gcPRO GigE Camera
            Lighting: EFFI-FD-200-200-000 High-Power Flat Light
    """

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
    Object_Cls.Visibility(False)

    # Return the object to the initialization position.
    Object_Cls.Reset()

    # Initialize the material class for randomizing material properties and handling 
    # baking operations.
    Material_Cls = Blender.Core.Material_Cls('UV_Scaled_128')

    # Generate random material properties and process bounding box detection.
    #   Note:
    #       If the returned information is None, it indicates that the bounding box 
    #       has not been generated for the material.
    #material_info_b_box = Material_Cls.Random('Area_Testing_Mat', list(Parameters.Scene.Basler_Cam_Str.Resolution.values()),
    #                                          obj_is_flipped, obj_center_px, Object_Cls.T.Get_Rotation('ZYX')[2])

    # Set the render file path.
    bpy.context.scene.render.filepath = f'{project_folder}/Data/Camera/Virtual/Image_{(CONST_INIT_INDEX):03}.png'

    # Check if the 'Composite' node exists in the node tree.
    if not bpy.context.scene.node_tree.nodes.get('Composite'):
        print('[WARNING] No Composite node found! Ensure that your compositor is correctly set up.')
    else:
        # Trigger render and save the image with compositing applied
        bpy.ops.render.render(animation=False, write_still=True)
        print(f'[INFO] Render completed and saved to: {bpy.context.scene.render.filepath}')

    # Release the classes.
    del Camera_Cls, Object_Cls, Material_Cls

if __name__ == '__main__':
    main()
