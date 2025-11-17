# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Typing (Support for type hints)
import typing as tp
# Matplotlib for debuging
import matplotlib.pyplot as plt

import math
import os

class Measure_Tools_Cls:

    def Get_Alpha_Beta_Parameters(image: tp.List[tp.List[int]], clip_limit: float) -> tp.Tuple[float, float]:
        """
        Souce:
        https://github.com/rparak/Object_Detection_Synthetic_Data/blob/main/src/Lib/Utilities/Image_Processing.py

        Description:
            Function to adjust the contrast and brightness parameters of the input image by clipping the histogram.

            The main core of the function is to obtain the alpha and beta parameters
            to determine the equation:
                g(i, j) = alpha * f(i, j) + beta,

                where g(i, j) are the input (source) pixels of the image, f(i, j) are the output pixels
                of the image, and alpha, beta are the contrast and brightness parameters.

        Args:
            (1) image [Vector<float> Image Shape {Resolution<x, y>}]: Input raw image.
            (2) clip_limit [float]: Parameter for histogram clipping in percentage.

        Returns:
            (1) parameter [float]: A gain (contrast) parameter called alpha.
            (2) parameter [float]: A bias (brightness) parameter called beta.
        """

        image_copy = image.copy()

        # Calculate the grayscale histogram of the image.
        image_hist = cv2.calcHist([image_copy], [0], None, [256], [0, 256])

        # Get the cumulative sum of the elements (histogram).
        c_hist = np.cumsum(image_hist)

        # Modify the percentage to clip the histogram.
        c_hist_max = c_hist[-1]
        clip_limit_mod = clip_limit * (c_hist_max/float(100.0*2.0))

        # Clip the histogram if the values are outside the limit.
        min_value = 0; max_value = c_hist.size - 1
        for _, c_hist_i in enumerate(c_hist):
            if c_hist_i < clip_limit_mod:
                min_value += 1

            if c_hist_i >= (c_hist_max - clip_limit_mod):
                max_value -= 1

        # Express the alpha and beta parameters.
        #   Gain (contrast) parameter.
        alpha = 255 / (max_value - min_value)
        #   Bias (brightness) parameter.
        beta  = (-1) * (min_value * alpha)

        return (alpha, beta)

    def Get_HSV_Image(input_image, mode = 'BACK_SIDE_CIRCLES_DETECTION'):
        '''
        Descritpion:
            Function sets up predefined HSV limitis acording to modes.

            Modes:
                - BACK_SIDE_CIRCLES_DETECTION
                - BACK_SIDE_HOLE_MEASURE
                - BACK_SIDE_RECTANGLE
                - FRONT_SIDE_RECTANGLE
                - FRONT_SIDE_HOLE_OUTTER

        Returns:
            Image with aplied HSV limits         
        '''
        processed_image = input_image.copy()
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)

        # Default declaration
        hue_min = 0
        hue_max = 179
        sat_min = 0
        sat_max = 255
        val_min = 0
        val_max = 255

        # Preprocess parameters of input image for localization of circles on backside
        if mode == 'BACK_SIDE_CIRCLES_DETECTION':
            # processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
            hue_min = 0; hue_max = 179; sat_min = 10; sat_max = 70; val_min = 190; val_max = 255

        # Preprocess of input image for measurement of circles on backside
        if mode == 'BACK_SIDE_HOLE_MEASURE':
            hue_min = 1; hue_max = 179; sat_min = 0; sat_max = 255; val_min = 0; val_max = 122

        if mode == 'BACK_SIDE_RECTANGLE':
            hue_min = 1; hue_max = 179; sat_min = 0; sat_max = 255; val_min = 0; val_max = 52

        # Front side (shiny) shape 
        if mode == 'FRONT_SIDE_RECTANGLE':
            hue_min = 1; hue_max = 179; sat_min = 0; sat_max = 255; val_min = 67; val_max = 255            


        # Apply trasformation
        lower_bound = (hue_min, sat_min, val_min)
        upper_bound = (hue_max, sat_max, val_max)
        mask = cv2.inRange(processed_image, lower_bound, upper_bound)

        return mask

    def Apply_Morphology_Filter(input_image, mode = 'OPEN_CLOSE', iters = 2):

        processed_image = input_image.copy()

        # Morphology kernels
        kernel_opening = np.ones((4,4),np.uint8)
        kernel_closing = np.ones((4,4),np.uint8)

        if mode =='OPEN' or mode =='OPEN_CLOSE':
            processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN,kernel_opening, iterations = iters)


        if mode =='CLOSE' or mode =='OPEN_CLOSE':
            processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE,kernel_closing, iterations = iters)

        return processed_image
    
    def Rotate_Point(point, center, angle_degrees):
        """
        Rotates a point (x, y) around a center point by a specified angle.

        Args:
            x (float): X-coordinate of the original point.
            y (float): Y-coordinate of the original point.
            center (tuple): (a, b) tuple representing the rotation center.
            angle_degrees (float): Rotation angle in degrees (positive = counter-clockwise).

        Returns:
            (float, float): Rotated point coordinates (x_n, y_n).
        """
        x, y = point
        a, b = center
        angle_rad = math.radians(angle_degrees)

        # Translate point to origin
        x_shifted = x - a
        y_shifted = y - b

        # Rotate point
        x_rotated = x_shifted * math.cos(angle_rad) - y_shifted * math.sin(angle_rad)
        y_rotated = x_shifted * math.sin(angle_rad) + y_shifted * math.cos(angle_rad)

        # Translate back
        x_n = x_rotated + a
        y_n = y_rotated + b

        return x_n, y_n

    def Get_Angle_Of_Rectangle(rect):

        # rect = cv2.minAreaRect(contour)
        # center, size, angle = rect
        box = cv2.boxPoints(rect)
        box = np.array(box)

        edge1 = box[1] - box[0]
        edge2 = box[2] - box[1]

        # Select the shorter edge
        edge = edge1 if np.linalg.norm(edge1) < np.linalg.norm(edge2) else edge2

        # Calculate the angle in radians and convert to degrees
        theta = math.degrees(math.atan2(edge[1], edge[0]))

        # Convert to range 0-360
        # angle_360 = theta % 360

        return theta

class Measure_Helpers_Cls:
    '''
    Contains debuging tools
    '''

    def Plot_HSV_Histograms(input_image):

        h, s, v = cv2.split(input_image)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("HSV Histograms")

        # Hue
        axes[0].hist(h.ravel(), bins=180, range=(0, 180), color='r', alpha=0.7)
        axes[0].set_title('Hue')
        axes[0].set_xlabel('H Value')
        axes[0].set_ylabel('Pixel Count')

        # Saturation
        axes[1].hist(s.ravel(), bins=256, range=(0, 255), color='g', alpha=0.7)
        axes[1].set_title('Saturation')
        axes[1].set_xlabel('S Value')

        # Value
        axes[2].hist(v.ravel(), bins=256, range=(0, 255), color='b', alpha=0.7)
        axes[2].set_title('Value')
        axes[2].set_xlabel('V Value')

        plt.tight_layout()
        plt.show()

        return fig, axes

    def Save_Image(image, filename):

        image_to_save = image.copy()

        # Define the file path where the image will be saved
        file_path = f'temp'

        # Ensure that the directory exists
        os.makedirs(file_path, exist_ok=True)  # If the directory does not exist, create it

        # Construct the complete image path (including filename)
        img_path_out = os.path.join(file_path, filename +'.png')

        # Save the image using cv2.imwrite()
        cv2.imwrite(img_path_out, cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))

        # Print the success message
        print(f'Image successfully saved at {img_path_out}')


class Measured_Object_Cls:

    def __init__(self, RATIO_X = 0.09744):
        self.PIXEL_TO_MM_X = RATIO_X
        self.PIXEL_TO_MM_Y = RATIO_X
        # Center of found object (rectangle) in given picture - (x , y) [px]
        self.center = None
        # Rotation angle of found object. Used for expected ROI of holes [deg]
        self.angle = None
        # Height of found object (rectangle) [px]
        self.height = None
        # Width of found object (rectangle) [px]
        self.width = None

        # Expected coordinates od holes [mm]
        self.hole_top_mm = (15, 17.5)
        self.hole_bottom_mm = (15, 42.5)
        # Expected coordinates converted to [px]
        self.hole_top_px = ((self.hole_top_mm[0] / self.PIXEL_TO_MM_X), (self.hole_top_mm[1] / self.PIXEL_TO_MM_Y))
        self.hole_bottom_px = (int(self.hole_bottom_mm[0] / self.PIXEL_TO_MM_X), int(self.hole_bottom_mm[1] / self.PIXEL_TO_MM_Y))

        # Square ROI parameters for holes localization
        self.Hole_ROI = 85 # half of a size of square 

    def Get_Backside_Size(self, input_image, draw = False):

        # width; height; angle = 0;0;0
        RATIO_X = 0.09744
        image_processed = input_image.copy()

        # Adjust Contrast
        ALPHA, BETA = Measure_Tools_Cls.Get_Alpha_Beta_Parameters(image_processed, 0.75)
        img_contrasted = cv2.convertScaleAbs(image_processed, alpha=ALPHA, beta=BETA)
        
        # HSV Analys
        image_rectangle_hsv = Measure_Tools_Cls.Get_HSV_Image(img_contrasted, 'BACK_SIDE_RECTANGLE')

        # Dilate + Blur
        kernel_dilate = np.ones((2,2),np.uint8)
        image_rectangle_blur_preprocess= cv2.GaussianBlur(image_rectangle_hsv, (5, 5), cv2.BORDER_DEFAULT)
        ret, image_rectangle_thresh_hsv = cv2.threshold(image_rectangle_blur_preprocess,50,255,cv2.THRESH_BINARY )
        image_dilated = cv2.dilate(image_rectangle_thresh_hsv, kernel_dilate, iterations=3)
        image_rectangle_hsv_invert = cv2.bitwise_not(image_dilated)
        image_rectangle_hsv_blur = cv2.GaussianBlur(image_rectangle_hsv_invert, (5, 5), cv2.BORDER_DEFAULT)

        # Canny edge detection
        image_canny_edges = cv2.Canny(image_rectangle_hsv_blur, 100, 200)

        # Find Lines
        lines = cv2.HoughLinesP(
            image_canny_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=15
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(img_contrasted, (x1, y1), (x2, y2), (0, 255, 0), 2)

        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append((x1, y1))
            points.append((x2, y2))

        points = np.array(points, dtype=np.float32)

        if len(points) >= 4:
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            if draw:
                cv2.drawContours(input_image, [box], 0, (0, 0, 255), 2)

            self.center = rect[0]
            w, h = rect[1]
            if h < w:
                w, h = h, w
            self.width, self.height = w, h
            # Get real angle in 0-360 deg
            self.angle = Measure_Tools_Cls.Get_Angle_Of_Rectangle(rect)
            # Angle of rectangle 0-90 deg 
            rect_angle = rect[2]

        return (self.width * RATIO_X , self.height * RATIO_X, rect_angle)
        # return image_canny_edges

    def __Measure_Hole(self, image_processed, x, y, expected_diameter, Hole_Type_Str , draw):
        '''
        # Adjusts Contrast, Brightness, HSV in ROI defined by parameters:
            # - x
            # - y
            # - self.Hole_ROI       

        Then deperforms canny edge a finds circle in threshed image.
            Returns radius of fond circle in [px]         
        '''
        # Get ROI with hole
        current_ROI = image_processed[int(y-self.Hole_ROI ) : int(y + self.Hole_ROI ), int(x - self.Hole_ROI ): int(x + self.Hole_ROI )]

        # Contrast
        ALPHA, BETA = Measure_Tools_Cls.Get_Alpha_Beta_Parameters(current_ROI, 0.75)
        image_contrasted = cv2.convertScaleAbs(current_ROI, alpha=ALPHA, beta=BETA)

        # Get HSV Analys
        image_hsv = Measure_Tools_Cls.Get_HSV_Image(image_contrasted, Hole_Type_Str)

        # Aply filter
        image_blur = cv2.GaussianBlur(image_hsv, (5, 5), cv2.BORDER_DEFAULT)

        # Threshold
        ret, thresh_image = cv2.threshold(image_blur,100,255,cv2.THRESH_BINARY )
        thresh_image = cv2.convertScaleAbs(thresh_image, alpha=1.2, beta=120)

        # Inverse image
        thresh_image = cv2.bitwise_not(thresh_image)

        # Canny edge detection
        image_circle = cv2.Canny(thresh_image, 100, 200)

        # Circles detection
        diameter_tolerance = 2
        circle = cv2.HoughCircles(image_circle, 
                                            cv2.HOUGH_GRADIENT , 
                                            dp=1.2, 
                                            minDist=50, 
                                            param1=50, 
                                            param2=22, 
                                            minRadius= int( (expected_diameter - diameter_tolerance) / (2* self.PIXEL_TO_MM_X )) , 
                                            maxRadius =  int( (expected_diameter + diameter_tolerance) / (2* self.PIXEL_TO_MM_X )))
        
        circle_in_hole = image_circle.copy()
        circle_in_hole= cv2.cvtColor(circle_in_hole, cv2.COLOR_GRAY2BGR)

        largest_radius = 0

        if circle is not None:
            circle = np.uint16(np.around(circle))
            for a in circle[0, :]:
                # Finds largest circle
                if a[2] > largest_radius:
                    circle_radius = a[2]
                    circle_center = (a[0], a[1])
                    largest_radius = a[2] 

            # Draw edges to original view
            if draw:
                delta_x = int(x) + circle_center[0] - self.Hole_ROI
                delta_y = int(y) + circle_center[1] - self.Hole_ROI
                cv2.circle(image_processed, (delta_x, delta_y), circle_radius , (0, 255, 255), 2)
        else:
            return 0, image_contrasted 

        return circle_radius, image_processed


    def Get_Backside_Holes(self, input_image, draw = False):
        '''
        Description:
            Function detects large holes on back ("dirty") side of object.
        
        Args:
            (1) input_image - source raw image, where holes are detected
            (2) draw - draws hole edges into retruned image

        Returns:
            (1) image_processed - copy of input image, modified depending on (2) input Argumet
            (2) top hole diameter in [mm]
            (3) bottom hole diameter in [mm]
        '''

        # Gets coordinates of expected hole location rotaded acording to angle detected in rectangel detection before
        #   - x1,y1 : top hole
        #   - x2,y2 : bottom hole
        angle_offset = 0
        search_done = False

        # Heuristic search for holes
        #   Description:
        #   -   The coordinates where the hole should be located are given in < self.hole_top_px, self.hole_bottom_px >
        #       If it is not located on them, the search is rotated 180 degrees - holes can be located on the other side because the object is rotated

        while search_done != True:

            image_processed = input_image.copy()

            x1, y1 = Measure_Tools_Cls.Rotate_Point((self.hole_top_px[0] + self.center[0] - self.width/2 , self.hole_top_px[1] + self.center[1] - self.height/2), self.center, self.angle + angle_offset)
            x2, y2 = Measure_Tools_Cls.Rotate_Point((self.hole_bottom_px[0] + self.center[0] - self.width/2, self.hole_bottom_px[1] + self.center[1] - self.height/2), self.center,  self.angle + angle_offset)

            expected_diameter = 6 #[mm]
            radius_top, img_top = self.__Measure_Hole(image_processed, x1, y1, expected_diameter, 'BACK_SIDE_HOLE_MEASURE', draw)
            radius_bottom, img_bot = self.__Measure_Hole(image_processed, x2, y2,expected_diameter, 'BACK_SIDE_HOLE_MEASURE', draw)

            if angle_offset == 0 and (radius_top == 0 or radius_bottom == 0):
                angle_offset = 180
            else:
                search_done = True
    
        return img_top , radius_top * 2 * self.PIXEL_TO_MM_X, radius_bottom  * 2 * self.PIXEL_TO_MM_X
    
    def Get_Frontside_Size(self, input_image, draw = False):
        image_processed = input_image.copy()

        # Adjust Contrast
        ALPHA, BETA = Measure_Tools_Cls.Get_Alpha_Beta_Parameters(image_processed, 0.75)
        img_contrasted = cv2.convertScaleAbs(image_processed, alpha=ALPHA, beta=BETA)
        # Measure_Helpers_Cls.Save_Image(img_contrasted, 'FrontSideProgress')
        
        # HSV Analys
        image_rectangle_blur_preprocess= cv2.GaussianBlur(img_contrasted, (5, 5), cv2.BORDER_DEFAULT)
        image_rectangle_hsv = Measure_Tools_Cls.Get_HSV_Image(image_rectangle_blur_preprocess, 'FRONT_SIDE_RECTANGLE')

        ret, thresh_image = cv2.threshold(image_rectangle_hsv,180,255,cv2.THRESH_BINARY )
        thresh_image = cv2.convertScaleAbs(thresh_image, alpha=1.2, beta=120)

        # Canny edge detection
        image_canny_edges = cv2.Canny(thresh_image, 100, 200)

        # Find Lines
        lines = cv2.HoughLinesP(
            image_canny_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=15
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # cv2.line(img_contrasted, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else: 
            return image_rectangle_blur_preprocess

        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append((x1, y1))
            points.append((x2, y2))

        points = np.array(points, dtype=np.float32)

        if len(points) >= 4:
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            if draw:
                cv2.drawContours(input_image, [box], 0, (0, 0, 255), 2)

            self.center = rect[0]
            w, h = rect[1]
            if h < w:
                w, h = h, w
            self.width, self.height = w, h
            # Get real angle in 0-360 deg
            self.angle = Measure_Tools_Cls.Get_Angle_Of_Rectangle(rect)
            # Angle of rectangle 0-90 deg 
            rect_angle = rect[2]

        return self.width * self.PIXEL_TO_MM_X , self.height * self.PIXEL_TO_MM_Y, rect_angle
    

    def Get_Frontside_Holes(self, input_image, hole_type_str, draw = False):
        '''
        Description:
            Function detects large holes on front (polished) side of object.
        
        Args:
            (1) input_image - source raw image, where holes are detected
            (2) hole_type_str - type of measured hole:
                                - OUTTER_HOLE
                                - INNER_HOLE
            (3) draw - draws hole edges into retruned image

        Returns:
            (1) image_processed - copy of input image, modified depending on (2) input Argumet
            (2) top hole diameter in [mm]
            (3) bottom hole diameter in [mm]
        '''

        image_processed = input_image.copy()

        if hole_type_str == 'OUTTER_HOLE':
            expected_diameter = 12  #[mm]
        elif hole_type_str == 'INNER_HOLE':
            expected_diameter = 6  #[mm]
        else:
            raise Exception("Invalid hole type selected.")

        # Gets coordinates of expected hole location rotaded acording to angle detected in rectangel detection before
        #   - x1,y1 : top hole
        #   - x2,y2 : bottom hole
        angle_offset = 0
        search_done = False

        # Heuristic search for holes
        #   Description:
        #   -   The coordinates where the hole should be located are given in < self.hole_top_px, self.hole_bottom_px >
        #       If it is not located on them, the search is rotated 180 degrees - holes can be located on the other side because the object is rotated
        
        while search_done != True:

            image_processed = input_image.copy()

            x1, y1 = Measure_Tools_Cls.Rotate_Point((self.hole_top_px[0] + self.center[0] - self.width/2 , self.hole_top_px[1] + self.center[1] - self.height/2), self.center, self.angle)
            x2, y2 = Measure_Tools_Cls.Rotate_Point((self.hole_bottom_px[0] + self.center[0] - self.width/2, self.hole_bottom_px[1] + self.center[1] - self.height/2), self.center,  self.angle)

            radius_top, img_top_circle = self.__Measure_Hole(image_processed, x1, y1, expected_diameter,  'FRONT_SIDE_RECTANGLE', draw )
            radius_bottom, img_bot_circle = self.__Measure_Hole(image_processed, x2, y2, expected_diameter, 'FRONT_SIDE_RECTANGLE', draw )

            if angle_offset == 0 and (radius_top == 0 or radius_bottom == 0):
                angle_offset = 180
            else:
                search_done = True

        return image_processed, radius_top * 2 * self.PIXEL_TO_MM_X, radius_bottom * 2 * self.PIXEL_TO_MM_X