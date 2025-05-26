import bpy
import random
import os
import numpy as np

'''
Call Get_Bounding_Box() method to randomize material and to get bounding box
'''


fingerprint_enabled = True
bake_image_name = "UV_Scaled_128"

def __generate_seed():
    '''
    Unused, it generated same value in every iteration
    '''
    #for i in range(10):
    seed_value = int.from_bytes(os.urandom(4), 'big')
    random.seed(seed_value)
    return seed_value

def randomize_material_values(material_name):
    """
    Randomizes specific value node outputs in the given material.
    """
    material = bpy.data.materials.get(material_name)
    if material is None:
        print(f"Material '{material_name}' not found.")
        return
    
    if material.use_nodes:
        for node in material.node_tree.nodes:
            if node.type == 'VALUE':
                if node.label == "location_z":
                    node.outputs[0].default_value = random.uniform(-100.0, 100.0)
                elif node.label == "seed":
                    node.outputs[0].default_value = random.uniform(-510.1, 5548.1)
                elif node.label == "scale_dots_x":
                    node.outputs[0].default_value = random.uniform(1.0, 3.3)
                elif node.label == "scale_dots_y":
                    node.outputs[0].default_value = random.uniform(0.2, 2.9)
                elif node.label == "roughness_1":
                    node.outputs[0].default_value = random.uniform(0.662, 1.0)
                elif node.label == "circle_top_x":
                    node.outputs[0].default_value = random.uniform(-0.4, 2.0)
                elif node.label == "circle_bot_x":
                    node.outputs[0].default_value = random.uniform(-0.4, 2.0)
                elif node.label == "enable":
                    global fingerprint_enabled
                    fingerprint_enabled = random.gauss(0.5, 0.2) > 0.5
                    print(f"Fingerprint enabled: {fingerprint_enabled}")
                    node.outputs[0].default_value = fingerprint_enabled
                elif node.label == "scale_dirty_color":
                    node.outputs[0].default_value = random.uniform(1, 15)
                elif node.label == "roughness_dirty_color":    
                    node.outputs[0].default_value = random.uniform(0.642, 1)
                elif node.label == "scale_dirty_roughness":     
                    node.outputs[0].default_value = random.uniform(1, 15)
                elif node.label == "detail_dirty_roughness_terrain":     
                    node.outputs[0].default_value = random.uniform(7.7, 15)                    
                elif node.label == "worn_strip_width":     
                    node.outputs[0].default_value = random.uniform(-0.05, -0.09)    
                elif node.label == "worn_strip_left_x":     
                    node.outputs[0].default_value = random.uniform(0.4, 0.35)    
                elif node.label == "worn_strip_right_x":     
                    node.outputs[0].default_value = random.uniform(-0.2 , -0.35)                    
                    
def __rewire_shader_and_bake(material_name):
    """
    Rewires the shader to bake the texture and restores the original connections afterward.
    """
    material = bpy.data.materials.get(material_name)
    if material is None or not material.use_nodes:
        print(f"Material '{material_name}' not found or does not use nodes.")
        return
    
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    mix_shader = next((node for node in nodes if node.type == 'MIX_SHADER' and node.label == "last_shader_mix"), None)
    material_output = next((node for node in nodes if node.type == 'OUTPUT_MATERIAL'), None)
    principled_bsdf = next((node for node in nodes if node.type == 'BSDF_PRINCIPLED' and node.label == "bake_output"), None)
    bake_image = next((node for node in nodes if node.type == 'TEX_IMAGE' and node.label == "bake_image"), None)
    
    if not all([mix_shader, material_output, principled_bsdf, bake_image]):
        print("Some required nodes were not found. Check labels.")
        return
    
    original_links = [link for link in links if link.to_node == material_output and link.to_socket.name == "Surface"]
    for link in original_links:
        links.remove(link)
    
    links.new(principled_bsdf.outputs[0], material_output.inputs["Surface"])
    
    bpy.context.view_layer.objects.active = bpy.context.object
    bake_image.select = True
    bpy.context.object.active_material = material
    bpy.context.object.active_material.node_tree.nodes.active = bake_image
    
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, use_selected_to_active=False)
    
    for link in original_links:
        links.new(mix_shader.outputs[0], material_output.inputs["Surface"])
    
    print("Baking completed, original shader links restored.")

def __get_bounding_box(image_name):
    """
    Detects the bounding box of the white area in the given image and outlines it in red.
    """
    image = bpy.data.images.get(image_name)
    if image is None:
        print(f"Image '{image_name}' not found.")
        return None
    
    width, height = image.size
    pixels = np.array(image.pixels).reshape((height, width, 4))
    
    min_x, max_x = width, 0
    min_y, max_y = height, 0
    
    for x in range(width):
        for y in range(height):
            if np.all(pixels[y, x, :3] > 0.05):  # Detect white areas
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
    
    if min_x < max_x and min_y < max_y:
        # Unused verifiaction box
        
        box_color = (1, 0, 0, 1)  # Red color for bounding box
        for x in range(min_x, max_x + 1):
            pixels[min_y, x] = box_color
            pixels[max_y, x] = box_color
        for y in range(min_y, max_y + 1):
            pixels[y, min_x] = box_color
            pixels[y, max_x] = box_color
        
        image.pixels = pixels.flatten()
        image.update()
        
        return min_x, max_x, min_y, max_y  
    else:
        print("No white area detected.")
        return None

def Get_Bounding_Box():
    '''
    Main function of material randomizer. Randomizes noises and fingerprint on object materials
    Output:
        - min_x, max_x, min_y, max_y    ... if fingerprint is generated
        - None                          ... if there is no fingerprint
    '''
    bounding_box = None
    # Execute the functions
    randomize_material_values("Area_Testing_Mat")
    if fingerprint_enabled:
        __rewire_shader_and_bake("Area_Testing_Mat")
        bounding_box = __get_bounding_box(bake_image_name)
        print(f"Bounding box coordinates: {bounding_box}")
    randomize_material_values("Dirty_Mat")
    randomize_material_values("Hole_Mill_Mat")
    return bounding_box

Get_Bounding_Box()