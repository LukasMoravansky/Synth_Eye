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

def main():
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

    P_extended = np.vstack((Camera_Cls.P(), np.ones(4)))
    p_tmp = (P_extended @ np.hstack((np.array(Object_Cls.T.p.all()), 1)))[0:-1]
    print(np.array(p_tmp/p_tmp[-1], dtype=int))
    print(bpy.data.objects[Object_Cls.Name].rotation_euler.z)
    
    # Release the classes.
    del Camera_Cls, Object_Cls

if __name__ == '__main__':
    main()
