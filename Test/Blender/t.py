import bpy

# x = 0.0075
# y = -0.05

# Deselect all objects before starting
for obj in bpy.context.selected_objects:
    obj.select_set(False)
bpy.context.view_layer.update()

# Clear any existing animation data
for obj in bpy.data.objects:
    obj.animation_data_clear()

# Example input data (position, rotation, and frame number)
# Each entry in the list is a tuple: (frame, position (x, y, z), rotation (x, y, z))
trajectory_data = [
    (1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (100, (0.0, 0.0, 0.5), (0.0, 0.0, 1.5708)),
    (200, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]

# Specify the object name to set the trajectory for
obj_name = "Basler"  # Replace with your object's name

# Get the object from the scene
obj = bpy.data.objects.get(obj_name)

if not obj:
    raise ValueError(f"Object '{obj_name}' not found!")

# Set the trajectory based on the input data
for frame, position, rotation in trajectory_data:
    # Set the object's location and rotation for the specified frame
    obj.location = position
    obj.rotation_euler = rotation  # Rotation in Euler angles (radians)
    
    # Ensure the object updates with the new location and rotation
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)

# Optionally, set the frame range for the animation
start_frame = min([f[0] for f in trajectory_data])
end_frame = max([f[0] for f in trajectory_data])
bpy.context.scene.frame_start = start_frame
bpy.context.scene.frame_end = end_frame

# Optionally, set playback range
bpy.context.scene.frame_current = start_frame
