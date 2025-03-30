import bpy



MATERIAL_NAME = "Area_Testing_Mat"


def get_or_create_material(material_name):
    # Zkontroluj, zda materiál existuje
    if material_name in bpy.data.materials:
        material = bpy.data.materials[material_name]
    else:
        # Pokud neexistuje, vytvoř nový materiál
        material = bpy.data.materials.new(name=material_name)
    
    # Nastav materiál jako aktivní
    if bpy.context.object and bpy.context.object.active_material != material:
        bpy.context.object.active_material = material

    return material

get_or_create_material(MATERIAL_NAME)