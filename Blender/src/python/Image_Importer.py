import bpy
import numpy as np
import math

scale_factor = 128 / (60 - (40 - 0.5 * 2))

def rotate_resize_and_insert(image_name, new_width, new_height, center_x, center_y, angle_degrees):
    # Load the original image
    image = bpy.data.images.get(image_name)
    if not image:
        print("Image not found!")
        return

    old_width, old_height = image.size
    pixels = np.array(image.pixels[:]).reshape((old_height, old_width, 4))  # RGBA

    # Calculate the scaling factor to maintain aspect ratio
    scaled_width = int(old_width * scale_factor)
    scaled_height = int(old_height * scale_factor)

    # Convert angle to radians
    angle = math.radians(angle_degrees)

    # Create a new image
    new_image = bpy.data.images.new(name=image_name + "_transformed", width=new_width, height=new_height, alpha=True)
    new_pixels = np.zeros((new_height, new_width, 4), dtype=np.float32)

    # Center of the old image in its own coordinates
    cx_old, cy_old = old_width / 2, old_height / 2

    # Center of the transformed image on the new canvas
    cx_new, cy_new = center_x, center_y  

    # Iterate through each pixel in the new image to compute the source color
    for y in range(new_height):
        for x in range(new_width):
            # Transform coordinates (reverse rotation and scaling)
            dx = (x - cx_new) / scale_factor
            dy = (y - cy_new) / scale_factor
            src_x = cx_old + dx * math.cos(angle) - dy * math.sin(angle)
            src_y = cy_old + dx * math.sin(angle) + dy * math.cos(angle)

            # Check if the coordinates are inside the original image
            if 0 <= src_x < old_width - 1 and 0 <= src_y < old_height - 1:
                # Bilinear interpolation
                x0, y0 = int(src_x), int(src_y)
                x1, y1 = min(x0 + 1, old_width - 1), min(y0 + 1, old_height - 1)
                dx, dy = src_x - x0, src_y - y0

                pixel00 = pixels[y0, x0]
                pixel01 = pixels[y0, x1]
                pixel10 = pixels[y1, x0]
                pixel11 = pixels[y1, x1]

                new_pixel = (pixel00 * (1 - dx) * (1 - dy) +
                             pixel01 * dx * (1 - dy) +
                             pixel10 * (1 - dx) * dy +
                             pixel11 * dx * dy)

                new_pixels[y, x] = new_pixel  # Store the interpolated pixel

    # Write new pixels to Blender
    new_image.pixels = new_pixels.ravel().tolist()  # Blender requires a list, not a NumPy array
    new_image.file_format = 'PNG'  # Set the file format
    print(f"Rotated and resized image '{new_image.name}' has been created.")

# Run the function - set new dimensions, center, and rotation angle
rotate_resize_and_insert("UV_Scaled_128", 1920, 1200, 512, 512, 10)