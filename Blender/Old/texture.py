import bpy
import numpy as np

# Set the resolution to match the camera output
width, height = 1920, 1200  # Camera resolution
noise_image = bpy.data.images.new('NoiseTexture', width=width, height=height)

# Generate random noise for the image with more contrast for grainy effect
mean = 0.0  # Darker mean value to simulate more grain in the image
stddev = 0.5  # Standard deviation (more variation for grain)

# Generate random noise (0 to 1, with increased contrast)
noise_data = np.random.normal(mean, stddev, (height, width))  # Mean = 0, stddev = 0.5
noise_data = np.clip(noise_data, 0, 1)  # Ensure values are between 0 and 1

# Convert noise to a format suitable for Blender image
# Blender uses RGBA (4 values per pixel: R, G, B, A)
noise_pixels = np.zeros((height, width, 4))  # Initialize RGBA array

# Set RGB channels to the noise values, and A (alpha) channel to 1 (fully opaque)
noise_pixels[..., 0] = noise_data  # R channel
noise_pixels[..., 1] = noise_data  # G channel
noise_pixels[..., 2] = noise_data  # B channel
noise_pixels[..., 3] = 1  # A channel (fully opaque)

# Flatten the noise_pixels array and assign it to the Blender image
noise_image.pixels = noise_pixels.flatten()

# Add the generated image to the texture slot
texture = bpy.data.textures.new('NoiseTexture', type='IMAGE')
texture.image = noise_image

# Print confirmation
print("Grainy noise texture 'NoiseTexture' has been created and added.")
