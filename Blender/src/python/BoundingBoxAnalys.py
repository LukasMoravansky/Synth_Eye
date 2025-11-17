import bpy
import numpy as np

# Parametry nového obrázku
new_image_name = "BakeTest_BoundingBox"
box_color = (1, 0, 0, 1)  # Červená barva obdélníku (RGBA)
bg_color = (0, 0, 0, 0)   # Průhledné pozadí (RGBA)

# Načtení původního obrázku
image_name = "BakeTest"
image = bpy.data.images.get(image_name)

if image is None:
    print(f"Obrázek '{image_name}' nebyl nalezen.")
else:
    # Načtení rozměrů obrázku
    width, height = image.size
    pixels = np.array(image.pixels).reshape((height, width, 4))  # (height, width, RGBA)

    # Inicializace hranic bílé šmouhy
    min_x, max_x = width, 0
    min_y, max_y = height, 0

    # Najdeme bílou šmouhu a její hranice
    for x in range(width):
        for y in range(height):
            if np.all(pixels[y, x, :3] == 1):  # Pokud je pixel bílý (RGB = 1,1,1)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

    if min_x < max_x and min_y < max_y:
        print(f"Obdélník kolem bílé šmouhy: ({min_x}, {min_y}) -> ({max_x}, {max_y})")

        # Vytvoření nového obrázku
        new_image = bpy.data.images.new(new_image_name, width=width, height=height, alpha=True)
        
        # Inicializace nových pixelů průhledným pozadím
        new_pixels = np.full((height, width, 4), bg_color)  # Průhledné RGBA pozadí

        # Kopírování bílé šmouhy do nového obrázku
        for x in range(width):
            for y in range(height):
                if np.all(pixels[y, x, :3] == 1):  # Pokud je pixel bílý
                    new_pixels[y, x] = (1, 1, 1, 1)  # Zachováme bílou barvu

        # Nakreslení obdélníku kolem šmouhy
        for x in range(min_x, max_x + 1):
            new_pixels[min_y, x] = box_color  # Horní hrana
            new_pixels[max_y, x] = box_color  # Dolní hrana

        for y in range(min_y, max_y + 1):
            new_pixels[y, min_x] = box_color  # Levá hrana
            new_pixels[y, max_x] = box_color  # Pravá hrana

        # Aktualizace nového obrázku
        new_image.pixels = new_pixels.flatten()
        new_image.update()
        print(f"Nový obrázek '{new_image_name}' vytvořen a obsahuje bílou šmouhu s obdélníkem.")
    else:
        print("Bílá šmouha nebyla nalezena.")
