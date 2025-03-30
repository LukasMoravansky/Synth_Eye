## =========================================================================== ## 
# MIT License
# Copyright (c) 2025 Roman Parak
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
## =========================================================================== ## 
# Author   : Roman Parak
# Email    : Roman.Parak@outlook.com
# Github   : https://github.com/rparak
# File Name: Core.py
## =========================================================================== ##

# BPY (Blender as a python)
import bpy
# System (Default)
import sys
sys.path.append('..')
# Numpy (Array computing)
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Blender/Core & Utilities
import Blender.Core
import Blender.Utilities
#   ../Parameters/Camera & Object
import Parameters.Scene
import Parameters.Object
#   ../Transformation/Utilities
import Transformation.Utilities.Mathematics as Mathematics
#   ../Transformation/Core
import Transformation.Core as Transformation

class Camera_Cls(object):
    """
    Description:
        A class for working with a camera and lighting in a Blender scene.

        The main part is to solve the equation:

            P = K x [R | t],
        
        where {K} is the instrict matrix that contains the intrinsic parameters, and {[R | t]} is the extrinsic 
        matrix that is combination of rotation matrix {R} and a translation vector {t}.

    Initialization of the Class:
        Args:
            (1) Cam_Param_Str [Camera_Parameters_Str]: The structure of the main parameters of the camera object.
                                                       Note:
                                                        See the ../Parameters/Scene.py script.
            (2) Ligh_Param_Str [Light_Parameters_Str]: The structure of the main parameters of the light object.
                                                       Note:
                                                        See the ../Parameters/Scene.py script.         
            (3) image_format [string]: The format to save the rendered image.

        Example:
            Initialization:
                # Assignment of the variables.
                image_format = 'PNG'

                # Initialization of the class.
                Cls = Camera_Cls(Camera_Parameters_Str(..), Light_Parameters_Str(..), 
                                 image_format)

            Features:
                # Properties of the class.
                None

                # Functions of the class.
                Cls.K; Cls.R_t
                    ...
                Cls.P
    """
        
    def __init__(self, Cam_Param_Str: Parameters.Scene.Camera_Parameters_Str, Ligh_Param_Str: Parameters.Scene.Light_Parameters_Str, 
                 image_format: str = 'PNG') -> None:
        try:
            assert Blender.Utilities.Object_Exist(Cam_Param_Str.Name) == True and Blender.Utilities.Object_Exist(Ligh_Param_Str.Name) == True
            
            # << PRIVATE >> #
            # The structure of the main parameters of the camera and light object.
            self.__Cam_Param_Str = Cam_Param_Str; self.__Light_Param_Str = Ligh_Param_Str
            # The format to save the rendered image.
            self.__image_format = image_format

            # Set the main parameters of the camera and lighting.
            self.__Set_Camera_Parameters(); self.__Set_Light_Parameters()

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            if Blender.Utilities.Object_Exist(Cam_Param_Str.Name) == False:
                print(f'[ERROR] The camera object named <{Cam_Param_Str.Name}> does not exist in the current scene.')
            if Blender.Utilities.Object_Exist(Ligh_Param_Str.Name) == False:
                print(f'[ERROR] The light object named <{Ligh_Param_Str.Name}> does not exist in the current scene.')  

    def __Update(self) -> None:
        """
        Description:
            Update the scene.
        """

        bpy.context.view_layer.update()

    def __Set_Camera_Parameters(self) -> None:
        """
        Description:
            Configure the main parameters of the camera object and the scene settings.
        """
       
        # Apply the camera's transformation matrix (position, rotation, scale).
        Blender.Utilities.Set_Object_Transformation(self.__Cam_Param_Str.Name, self.__Cam_Param_Str.T)
        self.__Update()

        # Adjust the sensor parameters of the camera: fit, width, and height.
        bpy.data.cameras[self.__Cam_Param_Str.Name].sensor_fit = self.__Cam_Param_Str.Sensor['Fit']
        bpy.data.cameras[self.__Cam_Param_Str.Name].sensor_width = self.__Cam_Param_Str.Sensor['Width']
        bpy.data.cameras[self.__Cam_Param_Str.Name].sensor_height = self.__Cam_Param_Str.Sensor['Height']
        
        # Set the camera's aperture F-Stop, which controls the amount of blur in the depth of field.
        bpy.data.cameras[self.__Cam_Param_Str.Name].dof.aperture_fstop = self.__Cam_Param_Str.Aperture_F_Stop
        
        # Enable and set the depth of field parameters (focus distance and DoF usage).
        bpy.data.cameras[self.__Cam_Param_Str.Name].dof.use_dof = self.__Cam_Param_Str.Use_DoF
        bpy.data.cameras[self.__Cam_Param_Str.Name].dof.focus_distance = self.__Cam_Param_Str.Focus_Distance
        
        # Set the resolution of the rendered image based on the given percentage and pixel dimensions.
        bpy.context.scene.render.resolution_percentage = self.__Cam_Param_Str.Resolution['Percentage']
        bpy.context.scene.render.resolution_x = self.__Cam_Param_Str.Resolution['x']
        bpy.context.scene.render.resolution_y = self.__Cam_Param_Str.Resolution['y']
        
        # Define the horizontal and vertical pixel aspect ratio (typically set to 1 for standard displays).
        bpy.context.scene.render.pixel_aspect_x = 1
        bpy.context.scene.render.pixel_aspect_y = 1
        
        # Set the camera's projection type (perspective, orthographic, etc.).
        bpy.data.cameras[self.__Cam_Param_Str.Name].type = self.__Cam_Param_Str.Type
        bpy.data.cameras[self.__Cam_Param_Str.Name].lens_unit = 'MILLIMETERS'
        
        # Convert exposure time from microseconds to seconds.
        __exposure_t_sec = self.__Cam_Param_Str.Exposure_Time / 1e6  

        # Enable motion blur.
        bpy.context.scene.render.use_motion_blur = True

        # Calculate Blender's motion blur shutter value.
        #   Note:
        #       The shutter value represents the fraction of a frame during which the virtual 
        #       shutter remains open.
        bpy.context.scene.render.motion_blur_shutter = Mathematics.Clamp(__exposure_t_sec * float(self.__Cam_Param_Str.FPS), 0.0, 1.0)

        # Simulate white balance adjustments using Blender's color management settings.
        bpy.context.scene.view_settings.look = 'None'
        
        # Set the gamma (brightness factor) for the scene, affecting the overall image brightness.
        bpy.context.scene.view_settings.gamma = 1.0
        
        # Enable curve mapping to allow for fine-grained adjustments in white balance.
        __curves_tmp = bpy.context.scene.view_settings
        __curves_tmp.use_curve_mapping = True
        #   Access the curve mapping settings
        __curve_map_tmp = __curves_tmp.curve_mapping
        #   Set the white balance using the provided RGB balance ratios.
        __curve_map_tmp.white_level = Blender.Utilities.Convert_Cam_To_Blender_Balance_Ratios(self.__Cam_Param_Str.Balance_Ratios)
   
        #   Apply the changes.
        __curve_map_tmp.update()
        
        # Set the image output format's color depth (8-bit by default).
        bpy.context.scene.render.image_settings.color_depth = '8'
        
        # Based on the specified spectrum type, choose the appropriate color mode (Monochrome or Color).
        if self.__Cam_Param_Str.Spectrum == 'Monochrome':
            bpy.context.scene.render.image_settings.color_mode = 'BW'
        elif self.__Cam_Param_Str.Spectrum == 'Color':
            bpy.context.scene.render.image_settings.color_mode = 'RGB'
        
        # Define the file format for saving the rendered image (e.g., PNG, JPEG, etc.).
        bpy.context.scene.render.image_settings.file_format = self.__image_format
        
        # Set the rendering device to GPU for faster rendering.
        bpy.context.scene.cycles.device = 'GPU'
        
        # Configure rendering parameters, including adaptive sampling for better performance.
        if bpy.context.scene.cycles.use_adaptive_sampling == False:
            bpy.context.scene.cycles.use_adaptive_sampling = True
        
        # Set the adaptive sampling threshold and the number of samples for noise reduction.
        bpy.context.scene.cycles.adaptive_threshold = 0.001
        bpy.context.scene.cycles.samples = 128
        
        # Set the frame rate for the rendered animation (frames per second).
        bpy.context.scene.render.fps = self.__Cam_Param_Str.FPS
        
        # Update the scene to apply the changes made to the camera and render settings.
        self.__Update()

    def __Set_Light_Parameters(self) -> None:
        """
        Description:
            Configure the main parameters of the light object within the scene.
        """

        # Apply the light's transformation matrix (position, rotation, and scale).
        Blender.Utilities.Set_Object_Transformation(self.__Light_Param_Str.Name, self.__Light_Param_Str.T)
        self.__Update()

        # Set the light's energy (intensity) in watts. This controls the overall brightness of the light.
        bpy.data.lights[self.__Light_Param_Str.Name].energy = self.__Light_Param_Str.Energy  

        # Assign the light's color using an RGB tuple, where each value is in the range [0, 1].
        bpy.data.lights[self.__Light_Param_Str.Name].color = self.__Light_Param_Str.Color  

        # Define the light's size, which affects the softness of shadows (especially for point and area lights).
        bpy.data.lights[self.__Light_Param_Str.Name].size = self.__Light_Param_Str.Size  

        # Set the light's shape. Options include: 'POINT', 'SUN', 'SPOT', 'AREA', or 'SQUARE'.
        bpy.data.lights[self.__Light_Param_Str.Name].shape = self.__Light_Param_Str.Shape

        # Ensure all the changes are reflected in the scene.
        self.__Update()

    def K(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the intrinsic matrix {K}, which contains the intrinsic parameters.

            Equation:
                K = [[alpha_u,   gamma, u_0],
                     [      0, alpha_v, v_0],
                     [      0,       0,   1]],

                where (alpha_u, alpha_v) are focal lengths expressed in units of pixels (note: usually the same), (u_0, v_0) are the principal 
                point (i.e. the central point of the image frame) and gamma is the skew between the axes (note: usually equal to zero).

        Returns:
            (1) parameter [Matrix<float> 3x3]: Instrict matrix of the camera.
        """

        try:
            assert bpy.data.cameras[self.__Cam_Param_Str.Name].sensor_fit == self.__Cam_Param_Str.Sensor['Fit']

            # Express the parameters of intrinsic matrix {K} of the camera.
            gamma = 0.0
            #   Focal lengths expressed in units of pixesl: alpha_u and alpha_v (note: alpha_v = alpha_u)
            alpha_u = (self.__Cam_Param_Str.f * self.__Cam_Param_Str.Resolution['x']) / bpy.data.cameras[self.__Cam_Param_Str.Name].sensor_width
            alpha_v = alpha_u
            #   Principal point: u_0 and v_0
            u_0 = self.__Cam_Param_Str.Resolution['x'] / 2.0
            v_0 = self.__Cam_Param_Str.Resolution['y'] / 2.0

            return Transformation.Homogeneous_Transformation_Matrix_Cls([[alpha_u,   gamma, u_0, 0.0],
                                                                         [    0.0, alpha_v, v_0, 0.0],
                                                                         [    0.0,     0.0, 1.0, 0.0],
                                                                         [    0.0,     0.0, 0.0, 1.0]], np.float64).R
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print('[ERROR] Incorrectly set method to fit the image and field of view angle inside the sensor.')
            print(f'[ERROR] The method must be set to {self.__Cam_Param_Str.Sensor["Fit"]}. Not to {bpy.data.cameras[self.__Cam_Param_Str.Name].sensor_fit}')

    def R_t(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the extrinsic matrix {[R | t]}, which is the combination of the rotation matrix {R} and a translation 
            vector {t}.

            The standard form of the homogeneous transformation matrix {T}:
                T = [R_{3x3}, t_{3x1}
                     0_{1x3}, 1_{1x1}],

                where R is a rotation matrix and t is a translation vector.

            The inverse form of the homogeneous transformation matrix:
                T^(-1) = [R^T_{3x3}, -R^T_{3x3} x t_{3x1}
                            0_{1x3},              1_{1x1}]

            The relationship between the extrinsic matrix parameters and the position 
            of the camera is:
                [R | t] = [R_{C} | C]^(-1),
                
                where C is a column vector describing the position of the camera center in world coordinates 
                and R_{C} is a rotation matrix describing the camera orientation.

            then we can express the parameters R, t as:
                R = R_{C}^T
                t = -R_{C}^T x C
            
        Returns:
            (1) parameter [Matrix<float> 3x4]: Extrinsic matrix of the camera.
        """
                
        # Modification matrix {R} to adjust the direction (sign {+, -}) of each axis.  
        R_mod = np.array([[1.0,  0.0,  0.0],
                          [0.0, -1.0,  0.0],
                          [0.0,  0.0, -1.0]], dtype=np.float64)

        # Expression of the parameters R, t of the extrinsic matrix.
        R = R_mod @ self.__Cam_Param_Str.T.Transpose().R
        t = (-1) * R @ self.__Cam_Param_Str.T.p.all()

        return np.hstack((R, t.reshape(3, 1)))

    def P(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the projection matrix {P} of the camera object.

            Equation:
                P = K x [R | t]

        Returns:
            (1) parameter [Matrix<float> 3x4]: Projection matrix of the camera.
        """
                
        return self.K() @ self.R_t()
    
    def Random(self) -> None:
        """
        Description:
            A function to randomly generate camera properties.

            Note:
                The function can be easily modified/extended with additional functions.
        """

        # Apply the additional noise to the image captured by the camera.
        #   adaptive_threshold = 0.001 .. 0.025
        bpy.context.scene.cycles.adaptive_threshold = np.float64(np.random.uniform(0.001, 0.025))

        #  Update the scene.
        self.__Update()

class Object_Cls(object):
    """
    Description:
        A class for working with a scanned object in a Blender scene.

    Initialization of the Class:
        Args:
            (1) Obj_Param_Str [Object_Parameters_Str]: The structure of the main parameters of the scanned object.
                                                       Note:
                                                        See the ../Parameters/Object script.
            (2) axes_sequence_cfg [string]: Rotation axis sequence configuration (e.g. 'ZYX', 'QUATERNION', etc.)

        Example:
            Initialization:
                # Assignment of the variables.
                Object_Str = Object_Parameters_Str()
                axes_sequence_cfg = 'ZYX'

                # Initialization of the class.
                Cls = Object_Cls(Object_Str, axes_sequence_cfg)

            Features:
                # Properties of the class.
                self.T; self.T_0
                    ...
                self.Bounding_Box

                # Functions of the class.
                Cls.Reset(); Cls.Visibility(True)
                    ...
                Cls.Random()
    """
    
    def __init__(self, Obj_Param_Str: Parameters.Object.Object_Parameters_Str, axes_sequence_cfg: str) -> None:
        try:
            assert Blender.Utilities.Object_Exist(Obj_Param_Str.Name) == True

            # << PRIVATE >> #
            self.__axes_sequence_cfg = axes_sequence_cfg
            # The structure of the main parameters of the scanned object.
            self.__Obj_Param_Str = Obj_Param_Str

            # Initialize the object's homogeneous transformation matrix.
            self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(self.__Obj_Param_Str.T.all().copy(), np.float64)

            # Create a dictionary and initialize the object's bounding box parameters.
            self.__Bounding_Box = {'Centroid': self.__T.p.all(), 'Size': self.__Obj_Param_Str.Bounding_Box.Size, 
                                   'Vertices': self.__Obj_Param_Str.Bounding_Box.Vertices.copy()}

            # Set the object transform to zero position.
            Blender.Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__T)
            self.__Update()

            # Return the object to the initialization position.
            self.Reset()
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] An object named <{Obj_Param_Str.Name}> does not exist in the current scene.')

    @property
    def Name(self) -> str:
        """
        Description:
            Get the name of the object.

        Returns:
            (1) parameter [string]: Name of the object.
        """

        return self.__Obj_Param_Str.Name

    @property
    def Id(self) -> int:
        """
        Description:
            Get the identification number of the object.

        Returns:
            (1) parameter [int]: Identification number.
        """

        return self.__Obj_Param_Str.Id
    
    @property
    def T_0(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the initial (null) homogeneous transformation matrix of an object.

        Returns:
            (1) parameter [Matrix<float> 4x4]: Homogeneous transformation matrix.
        """

        return self.__Obj_Param_Str.T
    
    @property
    def T(self):
        """
        Description:
            Get the actual homogeneous transformation matrix of an object.

        Returns:
            (1) parameter [Matrix<float> 4x4]: Homogeneous transformation matrix.
        """
                
        return self.__T
    
    @property
    def Vertices(self):
        """
        Description:
            Get the positions (x, y, z) of the vertices of the given object.

        Returns:
            (1) parameter [Matrix<float> 3xn]: The vertices of the scanned object.
                                               Note:
                                                Where n is the number of vertices.
        """
                
        return np.array(Blender.Utilities.Get_Vertices_From_Object(self.__Obj_Param_Str.Name),
                        dtype=np.float64)
    
    @property
    def Bounding_Box(self) -> tp.Tuple[tp.List[float], tp.List[float], tp.List[tp.List[float]]]:
        """
        Description:
            Get the main parameters of the object's bounding box.

        Returns:
            (1) parameter [Dictionary {'Centroid': Vector<float> 1x3, 'Size': Vector<float> 1x3, 
                                       'Vertices': Matrix<float> 3x8}]: The main parameters of the bounding box as a dictionary.
        """

        # Oriented Bounding Box (OBB) transformation according to the homogeneous transformation matrix of the object.
        q = self.__T.Get_Rotation('QUATERNION'); p = self.__T.p.all()
        for i, point_i in enumerate(self.__Obj_Param_Str.Bounding_Box.Vertices):
            self.__Bounding_Box['Vertices'][i, :] = q.Rotate(Transformation.Vector3_Cls(point_i, np.float64)).all() + p

        # The center of the bounding box is the same as the center of the object.
        self.__Bounding_Box['Centroid'] = p
        
        return self.__Bounding_Box
    
    def __Update(self) -> None:
        """
        Description:
            Update the scene.
        """

        bpy.context.view_layer.update()
    
    def Reset(self) -> None:
        """
        Description:
            Function to return the object to the initialization position.
        """

        self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(self.__Obj_Param_Str.T.all().copy(), np.float64)

        # Set the transformation of the object to the initial position.
        Blender.Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__T)
        self.__Update()

    def Visibility(self, state: bool) -> None:
        """
        Description:
            Function to enable and disable the visibility of an object.

        Args:
            (1) state [bool]: Enable (True) / Disable (False). 
        """

        Blender.Utilities.Object_Visibility(self.__Obj_Param_Str.Name, state)
        self.__Update()

    def Random(self) -> None:
        """
        Description:
            Function for random generation of object transformation (position, rotation). The boundaries of the random 
            generation are defined in the object structure.

            Note:
                If there are no boundaries in any axis (equals None), continue without generating.
        """

        # Initialize position and rotation arrays with zeros.
        p = np.zeros(3, np.float64); theta = p.copy()

        # Iterate over the items in the Position and Rotation dictionaries.
        for i, (position_item, rotation_item) in enumerate(zip(self.__Obj_Param_Str.Limit.Position.items(), 
                                                               self.__Obj_Param_Str.Limit.Rotation.items())):
            # Process position {p} if valid range is provided.
            if position_item[1] is not None:
                # Generate a random value within the specified range.
                p_tmp = np.random.uniform(position_item[1]['range'][0], position_item[1]['range'][1])
                
                # Use the generated value or convert it to binary based on the distribution flag.
                p[i] = p_tmp if position_item[1]['Use_Distribution'] else int(p_tmp > 0.5)
            
            # Process rotation {theta} if valid range is provided
            if rotation_item[1] is not None:
                # Generate a random value within the specified range for rotation.
                theta_tmp = np.random.uniform(rotation_item[1]['range'][0], rotation_item[1]['range'][1])
                
                # Use the generated value or convert it to binary based on the distribution flag.
                theta[i] = theta_tmp if rotation_item[1]['Use_Distribution'] else int(theta_tmp > 0.5)

        # Create a homogeneous transformation matrix from random values.
        self.__T = self.__Obj_Param_Str.T.Rotation(theta, self.__axes_sequence_cfg).Translation(p)

        # Set the object transformation.
        Blender.Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__T)
        self.__Update()



