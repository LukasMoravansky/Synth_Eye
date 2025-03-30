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
# File Name: Object.py
## =========================================================================== ##

# Numpy (Array computing)
import numpy as np
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Transformation/Core
import Transformation.Core as Transformation 
#   ../Transformation/Utilities
import Transformation.Utilities.Mathematics as Mathematics
#   ../Parameters/Primitives
import Utilities.Primitives as Primitives

@dataclass
class Limit_Str:
    """
    Description:
        An object structure that defines the boundaries of an object's position/rotation.

        Note:
            The individual parts of the structure shall have the following shape:
                {'x': {'range': [x_{-}, x_{+}], 'Use_Distribution': False}, 'y': {'range': [y_{-}, y_{+}], 'Use_Distribution': True}, 
                 'z': {'range': [z_{-}, z_{+}], 'Use_Distribution': False}},

                 where indicates that values will be distributed over the range.
            
            If the object does not have a boundary in the axis, just mark it as None. For example, if there 
            is no boundary in the 'y' axis, then:
                {'x': {'range': [x_{-}, x_{+}], 'Use_Distribution': False}, 'y': None, 
                 'z': {'range': [z_{-}, z_{+}], 'Use_Distribution': True}}
    """
        
    Position: tp.Dict[str, tp.Optional[tp.Union[tp.List[float], 
                                                tp.Dict[str, tp.Any]]]] = field(default_factory=dict)
    Rotation: tp.Dict[str, tp.Optional[tp.Union[tp.List[float], 
                                                tp.Dict[str, tp.Any]]]] = field(default_factory=dict)

@dataclass
class Object_Parameters_Str:
    """
    Description:
        The structure of the main parameters of the scanned object.
    """

    # The name of the object.
    Name: str = 'Object'

    # The identification number (ID) of the object.
    Id: int = 0

    # Homogeneous transformation matrix of the initial position of the object.
    T: tp.List[tp.List[float]] = field(default_factory=list)

    # Parameters of the object bounding box. Generated from the script gen_object_bounding_box.py. 
    Bounding_Box: Primitives.Box_Cls = field(default_factory=Primitives.Box_Cls)

    # The position/rotation boundaries of the object.
    Limit: Limit_Str = field(default_factory=Limit_Str)

"""
Description:
    The main parameters of the test object in the scene.
"""
Object_001_Str = Object_Parameters_Str(Name='Object_001', Id=0)

# Homogeneous transformation matrix {T} of the object.
Object_001_Str.T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float64).Rotation([0.0, 0.0, 0.0], 
                                                                                          'ZYX').Translation([0.0075, 0.0, 0.004])

# Parameters of the object bounding box.
Object_001_Str.Bounding_Box = Primitives.Box_Cls([0.0, 0.0, 0.0], [0.04, 0.06, 0.008])

# Position and rotation limits of the object.
Object_001_Str.Limit.Position = {'x': {'range': [-0.045, 0.045], 'Use_Distribution': True}, 
                                 'y': {'range': [-0.015, 0.015], 'Use_Distribution': True}, 'z': None}
Object_001_Str.Limit.Rotation = {'x': None, 'y': {'range': [0.0, Mathematics.CONST_MATH_PI], 'Use_Distribution': False}, 
                                 'z': {'range': [-Mathematics.CONST_MATH_HALF_PI, Mathematics.CONST_MATH_HALF_PI], 'Use_Distribution': True}}

