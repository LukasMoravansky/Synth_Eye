# BPY (Blender as a python)
import bpy

"""
Description:
    Initialization of constants.
"""
# Checkerboard inner corners (columns, rows).
CONST_CHECKERBOARD_INNER_CORNERS = (8, 11)
# Checkerboard square size in meters.
CONST_SQUARE_SIZE_M = 12.0 / 1000.0
# Checkerboard size (width, height) on A4 format in meters.
CONST_A4_SIZE_M = (0.210, 0.297)

def main():
    """
    Description:
        Generate the checkerboard squares on top of the base plane using given materials.
    """

    # Select and delete all objects in the current scene.
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)

    # Create a list of materials for the checkerboard.
    mat_checkerboard = []; colors = {'White': (1, 1, 1), 'Black': (0, 0, 0)}
    for name_i, color_i in colors.items():
        mat = bpy.data.materials.new(name_i)
        mat.diffuse_color = (*color_i, 1)
        mat_checkerboard.append(mat)

    # Create an A4-sized plane slightly below the checkerboard and assign material.
    #   Slightly below checkerboard.
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -0.001))
    base_plane = bpy.context.object
    #   Scale the size of the plane in half (the default plane is 2x2).
    base_plane.scale.x = CONST_A4_SIZE_M[0]
    base_plane.scale.y = CONST_A4_SIZE_M[1]
    base_plane.data.materials.append(mat_checkerboard[0])

    # Generate checkerboard squares within Blender environment.
    for x in range(CONST_CHECKERBOARD_INNER_CORNERS[0]):
        for y in range(CONST_CHECKERBOARD_INNER_CORNERS[1]):
            xpos = (x * CONST_SQUARE_SIZE_M) - (CONST_CHECKERBOARD_INNER_CORNERS[0] * CONST_SQUARE_SIZE_M) / 2 + CONST_SQUARE_SIZE_M / 2
            ypos = (y * CONST_SQUARE_SIZE_M) - (CONST_CHECKERBOARD_INNER_CORNERS[1] * CONST_SQUARE_SIZE_M) / 2 + CONST_SQUARE_SIZE_M / 2
            bpy.ops.mesh.primitive_plane_add(size=CONST_SQUARE_SIZE_M, location=(xpos, ypos, 0))
            square = bpy.context.object

            # Assign black or white material based on checkerboard pattern
            if (x + y) % 2 == 0:
                square.data.materials.append(mat_checkerboard[1])
            else:
                square.data.materials.append(mat_checkerboard[0])
        
if __name__ == '__main__':
    main()