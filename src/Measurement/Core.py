# OpenCV library for computer vision tasks
import cv2
# Numpy (Array computing)
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Library:
#   ../Utilities/General
import Utilities.General

class Measurement_Object_Parameters_Cls:
    # Create a global data type for the class.
    cls_data_type = tp.TypeVar('cls_data_type')

    def __init__(self, image, object_bounding_box, class_id):
        self.__image = image.copy()
        self.__cropped_image = self.__Get_Cropped_Image(object_bounding_box)
        self.__cls_id = class_id

    @property
    def Cropped_Image(self) -> cls_data_type:
        return self.__cropped_image.copy()
    
    def __Get_Cropped_Image(self, b_box):
        # Determine resolution of the processed image.
        img_h, img_w = self.__image.shape[:2]
        Resolution = {'x': img_w, 'y': img_h}

        # Converts bounding box coordinates from YOLO format to absolute pixel coordinates.
        abs_coordinates_obj = Utilities.General.YOLO_To_Absolute_Coordinates({'x_c': b_box[0], 'y_c': b_box[1], 
                                                                              'width': b_box[2], 'height': b_box[3]}, 
                                                                             Resolution)
            
        # Calculate object bounding box edges from center-based coordinates.
        obj_left = int(abs_coordinates_obj['x'] - abs_coordinates_obj['width'] / 2)
        obj_top = int(abs_coordinates_obj['y'] - abs_coordinates_obj['height'] / 2)
        obj_right = int(abs_coordinates_obj['x'] + abs_coordinates_obj['width'] / 2)
        obj_bottom = int(abs_coordinates_obj['y'] + abs_coordinates_obj['height'] / 2)

        # Crop the object region from the original image.
        return self.__image[obj_top:obj_bottom, obj_left:obj_right]
    
    def __Remove_Background(self):
        pass
    
    def Solve(self):
        pass

