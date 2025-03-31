def convert_inner_to_full_resolution_with_scaling_from_bbox(outer_bbox, inner_bbox, full_resolution, original_resolution):
    # Outer bounding box coordinates: [x_min, y_min, x_max, y_max]
    outer_x_min, outer_y_min, outer_x_max, outer_y_max = outer_bbox
    # Inner bounding box coordinates: [x_min, y_min, x_max, y_max]
    inner_x_min, inner_y_min, inner_x_max, inner_y_max = inner_bbox
    
    # Full image resolution (width, height)
    img_width, img_height = full_resolution
    # Original image resolution (width, height)
    orig_width, orig_height = original_resolution
    
    # Calculate the width and height of the outer bounding box in the original image
    outer_width = outer_x_max - outer_x_min
    outer_height = outer_y_max - outer_y_min
    
    # Calculate the scaling factors based on the outer bounding box size
    scale_x = img_width / outer_width
    scale_y = img_height / outer_height
    
    # Adjust the inner bounding box to the full image coordinates (with respect to the outer bounding box)
    adjusted_x_min = inner_x_min + outer_x_min
    adjusted_y_min = inner_y_min + outer_y_min
    adjusted_x_max = inner_x_max + outer_x_min
    adjusted_y_max = inner_y_max + outer_y_min
    
    # Apply the scaling to convert the adjusted coordinates to the full image resolution
    scaled_x_min = adjusted_x_min * scale_x
    scaled_y_min = adjusted_y_min * scale_y
    scaled_x_max = adjusted_x_max * scale_x
    scaled_y_max = adjusted_y_max * scale_y
    
    # Return the scaled bounding box in pixels for the full resolution image
    return [scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max]

# Example bounding boxes and resolutions
outer_bbox = [687, 95, 1239, 743]  # Coordinates of the outer bounding box in the original image
inner_bbox = [5, 48, 79, 127]  # Coordinates of the inner bounding box in the original image
full_resolution = (1920, 1200)  # Full image resolution (1920x1200)
original_resolution = (128, 128)  # Original image resolution (128x128)

# Get the scaled bounding box in the full image resolution (in pixels)
scaled_bbox = convert_inner_to_full_resolution_with_scaling_from_bbox(outer_bbox, inner_bbox, full_resolution, original_resolution)
print(scaled_bbox)
