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
# List of class IDs to be removed from label data.
CONST_CLS_ID_REMOVE = [2]
# Name of the output dataset where the processed images and labels will be saved.
CONST_DATASET_NAME = 'Dataset_v2'
# List of partitions to be processed.
CONST_PARTITION_LIST = ['train', 'valid', 'test']

def main():
    """
    Description:
        A program to generate processed images from a defined dataset by applying transformations, filtering 
        labels, and saving new images and labels.
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
        
            # Load image.
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f'Unable to load image from: {image_path}')

            # Apply the image processing pipeline depending on partition type.
            if partition_name_i in ['train', 'valid']:
                processed_image = Process_Synthetic_Image_Cls.Apply(image)
            else:
                processed_image = Process_Real_Image_Cls.Apply(image)

            # Save processed image.
            output_image_path = f'{output_image_dir}/{image_name}.png'
            cv2.imwrite(output_image_path, processed_image)

            # Process and save label data.
            output_label_path = f'{output_label_dir}/{image_name}'
            if is_background_flag:
                # Create an empty label file for background images.
                open(f'{output_label_path}.txt', 'w').close()
            else:
                # For non-background images, load original labels, filter or remap class IDs,
                # and save the updated labels in YOLO format.
                label_input_path = os.path.join(
                    project_folder, 'Data', 'Dataset_v1', 'labels', partition_name_i, f'{image_name}'
                )
                label_data_raw = File_IO.Load(label_input_path, 'txt', ' ')

                # Filter or remap labels based on class ID.
                label_data = []
                if label_data_raw.size > 0 and np.isin(CONST_CLS_ID_REMOVE, label_data_raw[:, 0]).any():
                    for label in label_data_raw:
                        if label[0] not in CONST_CLS_ID_REMOVE:
                            label_data.append(np.append([0], label[1:]))
                    label_data = np.array(label_data, dtype=label_data_raw.dtype)
                else:
                    new_class_id = 1 if label_data_raw[0, 0] == 1 else 0
                    label_data = np.array([np.append([new_class_id], label_data_raw[0, 1:])], dtype=label_data_raw.dtype)

                # Format and save label data.
                label_data[:, 0] = label_data[:, 0].astype(int)
                for label in label_data:
                    formatted_line = f'{int(label[0])} ' + ' '.join(f'{x:.6f}' for x in label[1:])
                    File_IO.Save(output_label_path, formatted_line.split(), 'txt', ' ')

        # Track partition count.
        partition_counts[partition_name_i] = len(image_dir_info)
        total_processed += len(image_dir_info)

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print('[INFO] Data generation completed successfully.')
    for partition, count in partition_counts.items():
        print(f'[INFO] [{partition.capitalize():<6} Partition] samples: {count}')
    print(f'[INFO] Total samples processed: {total_processed}')
    print(f'[INFO] Time: {int(minutes)}m {int(seconds)}s')

if __name__ == '__main__':
    sys.exit(main())