import bpy

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Parameters
CHECKERBOARD = (11 - 1, 8 - 1)  # inner corners (columns, rows)
SQUARE_SIZE_MM = 12.0           # square size in millimeters
A4_SIZE_M = (0.210, 0.297)      # A4 size in meters (width, height)

# Convert mm to meters
SQUARE_SIZE = SQUARE_SIZE_MM / 1000.0

cols, rows = CHECKERBOARD
num_squares_x = cols + 1
num_squares_y = rows + 1

# Create materials
def create_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)  # RGBA
    return mat

white_mat = create_material('White', (1, 1, 1))
black_mat = create_material('Black', (0, 0, 0))
base_mat = create_material('BaseWhite', (1, 1, 1))

# Create A4 base plane
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -0.001))  # slightly below checkerboard
base_plane = bpy.context.object
base_plane.scale.x = A4_SIZE_M[0] / 2  # scale halves plane size (default plane is 2x2)
base_plane.scale.y = A4_SIZE_M[1] / 2
base_plane.data.materials.append(base_mat)

# Create checkerboard squares
for x in range(num_squares_x):
    for y in range(num_squares_y):
        xpos = (x * SQUARE_SIZE) - (num_squares_x * SQUARE_SIZE) / 2 + SQUARE_SIZE / 2
        ypos = (y * SQUARE_SIZE) - (num_squares_y * SQUARE_SIZE) / 2 + SQUARE_SIZE / 2
        bpy.ops.mesh.primitive_plane_add(size=SQUARE_SIZE, location=(xpos, ypos, 0))
        square = bpy.context.object
        # Assign black or white material alternately
        if (x + y) % 2 == 0:
            square.data.materials.append(black_mat)
        else:
            square.data.materials.append(white_mat)
