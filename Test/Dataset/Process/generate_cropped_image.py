# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../../' + 'src' not in sys.path:
    sys.path.append('../../../' + 'src')
# OS (Operating system interfaces)
import os
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Numpy (Array computing)
import numpy as np
# Time (Time access and conversions)
import time
# Custom Library:
#   ../Utilities/File_IO
import Utilities.File_IO as File_IO
#   ../Utilities/Image_Processing
import Utilities.Image_Processing
#   ../Utilities/General
import Utilities.General

"""
Description:
    Initialization of constants.
"""
# List of class IDs to be preserved from label data.
CONST_CLS_ID_PRESERVE = [2]
# Name of the output dataset where the processed images and labels will be saved.
CONST_DATASET_NAME = 'Dataset_v3'
# List of partitions to be processed.
CONST_PARTITION_LIST = ['train', 'valid', 'test']

def main():
    """
    Description:
        A program that processes images by cropping around a primary object (class_id=0), applying class-specific 
        image transformations, remapping associated defect annotations (e.g., class_id=2), and saving the results 
        as cropped images with updated YOLO labels into a new organized dataset.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

    # Initialize the class for custom image processing.
    Process_Real_Image_Cls = Utilities.Image_Processing.Process_Image_Cls('real')
    Process_Synthetic_Image_Cls = Utilities.Image_Processing.Process_Image_Cls('synthetic')

    # Initialize the timer and counters.
    start_time = time.time()
    total_processed = 0; partition_counts = {}

    for partition_name_i in CONST_PARTITION_LIST:
        input_image_dir = os.path.join(project_folder, 'Data', 'Dataset_v1', 'images', partition_name_i)
        output_image_dir = os.path.join(project_folder, 'Data', CONST_DATASET_NAME, 'images', partition_name_i)
        output_label_dir = os.path.join(project_folder, 'Data', CONST_DATASET_NAME, 'labels', partition_name_i)

        # Ensure output directories exist.
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        # Extracts numeric identifiers and background flags from filenames in a specified directory.
        image_dir_info = Utilities.General.Extract_Num_From_Filename('Image', input_image_dir)

        for image_dir_info_i in image_dir_info:
            image_path = image_dir_info_i['Path']; image_name = image_dir_info_i['Name']
            is_background_flag = image_dir_info_i['Is_Background']
        
            # Skip processing for background images.
            if is_background_flag == True:
                continue

            # Load raw label data.
            label_input_path = os.path.join(
                project_folder, 'Data', 'Dataset_v1', 'labels', partition_name_i, f'{image_name}'
            )
            label_data_raw = File_IO.Load(label_input_path, 'txt', ' ')

            # Initialize object bounding of target object (class_id=0).
            object_bbox = None

            # Determine the bounding box of the object.
            for label_i in label_data_raw:
                class_id = int(label_i[0])
                if class_id == 0:
                    # YOLO format: [x_center, y_center, width, height].
                    object_bbox = label_i[1:5]
                    break

            if object_bbox is None:
                print(f'[INFO] No object with class_id=0 found in {image_name}, skipping crop.')
                continue

            # Load image.
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f'Unable to load image from: {image_path}')

            # Apply the image processing pipeline depending on partition type.
            if partition_name_i in ['train', 'valid']:
                processed_image = Process_Synthetic_Image_Cls.Apply(image)
            else:
                processed_image = Process_Real_Image_Cls.Apply(image)
            
            # Determine resolution of the processed image.
            img_h, img_w = processed_image.shape[:2]
            Resolution = {'x': img_w, 'y': img_h}

            # Filter defect bounding boxes with preserved class IDs.
            defect_bbox = []
            if np.isin(CONST_CLS_ID_PRESERVE, label_data_raw[:, 0]).any() and label_data_raw[:, 0].size >= 1:
                for _, label_data_i in enumerate(label_data_raw):
                    if label_data_i[0] in CONST_CLS_ID_PRESERVE:
                        defect_bbox.append(np.append([0], label_data_i[1:]))
                        break
            else:
                if label_data_raw[0, 0] == 0:
                    # Create an empty label file for background images.
                    open(f'{output_label_dir}/{image_name}.txt', 'w').close()
                else:
                     # Skip images without target or defect classes.
                    continue

            # Converts bounding box coordinates from YOLO format to absolute pixel coordinates.
            abs_coordinates_obj = Utilities.General.YOLO_To_Absolute_Coordinates({'x_c': object_bbox[0], 'y_c': object_bbox[1], 
                                                                                  'width': object_bbox[2], 'height': object_bbox[3]}, 
                                                                                   Resolution)
            
            # Calculate object bounding box edges from center-based coordinates.
            obj_left = int(abs_coordinates_obj['x'] - abs_coordinates_obj['width'] / 2)
            obj_top = int(abs_coordinates_obj['y'] - abs_coordinates_obj['height'] / 2)
            obj_right = int(abs_coordinates_obj['x'] + abs_coordinates_obj['width'] / 2)
            obj_bottom = int(abs_coordinates_obj['y'] + abs_coordinates_obj['height'] / 2)

            # Crop the object region from the original image.
            cropped_img = processed_image[obj_top:obj_bottom, obj_left:obj_right]
            cropped_h, cropped_w = cropped_img.shape[:2]

            # Update defect labels relative to cropped image.
            if len(defect_bbox) > 0:
                # Converts bounding box coordinates from YOLO format to absolute pixel coordinates.
                abs_coordinates_defect = Utilities.General.YOLO_To_Absolute_Coordinates({'x_c': defect_bbox[0][1::][0], 'y_c': defect_bbox[0][1::][1], 
                                                                                         'width': defect_bbox[0][1::][2], 'height': defect_bbox[0][1::][3]}, 
                                                                                          Resolution)
                # Shift defect coordinates relative to cropped image.
                new_x_abs = abs_coordinates_defect['x'] - obj_left
                new_y_abs = abs_coordinates_defect['y'] - obj_top

                # Skip if defect center is outside cropped image.
                if not (0 <= new_x_abs <= cropped_w and 0 <= new_y_abs <= cropped_h):
                    continue

                # Converts bounding box coordinates from absolute pixel values to YOLO format.
                yolo_coordinates_defect = Utilities.General.Absolute_Coordinates_To_YOLO(
                    {'x': new_x_abs, 'y': new_y_abs, 'width': abs_coordinates_defect['width'], 'height': abs_coordinates_defect['height']}, {'x': cropped_w, 'y': cropped_h}
                )

                # Clamp values to [0, 1] if needed.
                new_label = np.array([[0, yolo_coordinates_defect['x_c'], 
                                        yolo_coordinates_defect['y_c'], 
                                        yolo_coordinates_defect['width'], 
                                        yolo_coordinates_defect['height']]], dtype=label_data_raw.dtype)

                # Format and save label data.
                new_label[:, 0] = new_label[:, 0].astype(int)
                for _, new_label_i in enumerate(new_label):
                    formatted_line = f'{int(new_label_i[0])} ' + ' '.join(f'{x:.6f}' for x in new_label_i[1:])
                    File_IO.Save(f'{output_label_dir}/{image_name}', formatted_line.split(), 'txt', ' ')

            # Save processed image.
            output_image_path = f'{output_image_dir}/{image_name}.png'
            cv2.imwrite(output_image_path, cropped_img)

        # Track partition count.
        partition_counts[partition_name_i] = len(image_dir_info)
        total_processed += len(image_dir_info)

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print('[INFO] Data generation completed successfully.')
    for partition, count in partition_counts.items():
        print(f'[INFO] [{partition.capitalize():<1} Partition] samples: {count}')
    print(f'[INFO] Total samples processed: {total_processed}')
    print(f'[INFO] Time: {int(minutes)}m {int(seconds)}s')
    
if __name__ == '__main__':
    sys.exit(main())