import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import ttk

CONST_INIT_INDEX = 2
CONST_INIT_INDEX_N = 5001

project_folder = os.getcwd().split('Synth_Eye')[0] + 'Synth_Eye'

# Load and prepare the image
img_original = cv2.imread(f'{project_folder}/Data/Dataset_v1/images/test/Image_{CONST_INIT_INDEX_N:03}.png')
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
img_raw = cv2.imread(f'{project_folder}/Data/Dataset_v1/images/train/Image_{CONST_INIT_INDEX:03}.png')
img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
img_tmp = img_rgb.copy()

# Initialize interactive plotting
plt.ion()
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Function to apply both contrast/brightness and HSV modifications
def update_image(event=None):
    alpha = float(alpha_var.get())
    beta = float(beta_var.get())
    hue_shift = float(hue_var.get())
    sat_scale = float(sat_var.get())
    val_scale = float(val_var.get())

    # Step 1: Apply contrast and brightness
    img_cb = cv2.convertScaleAbs(img_tmp, alpha=alpha, beta=beta)

    # Step 2: Convert to HSV
    img_hsv = cv2.cvtColor(img_cb, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Modify HSV channels
    img_hsv[..., 0] = (img_hsv[..., 0] + hue_shift) % 180  # OpenCV Hue range: [0,179]
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * sat_scale, 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * val_scale, 0, 255)

    img_hsv = img_hsv.astype(np.uint8)

    # Convert back to RGB
    img_modified = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # Display
    axs[0].clear()
    axs[0].imshow(img_original)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].clear()
    axs[1].imshow(img_modified)
    axs[1].set_title('Modified Image (Contrast/Brightness + HSV)')
    axs[1].axis('off')

    fig.canvas.draw()
    fig.canvas.flush_events()

# Tkinter window
root = tk.Tk()
root.title("Real-Time Image Adjuster: Contrast, Brightness, HSV")

# GUI controls
def add_slider(label, variable, from_, to_, row):
    ttk.Label(root, text=label).grid(row=row, column=0, sticky='w')
    slider = ttk.Scale(root, from_=from_, to=to_, variable=variable, orient='horizontal', length=300, command=update_image)
    slider.grid(row=row, column=1)
    return slider

alpha_var = tk.DoubleVar(value=1.0)
beta_var = tk.DoubleVar(value=0.0)
hue_var = tk.DoubleVar(value=0.0)
sat_var = tk.DoubleVar(value=1.0)
val_var = tk.DoubleVar(value=1.0)

add_slider("Contrast (alpha)", alpha_var, 0.0, 10.0, 0)
add_slider("Brightness (beta)", beta_var, -200.0, 200.0, 1)
add_slider("Hue Shift (°)", hue_var, -90.0, 90.0, 2)
add_slider("Saturation Scale", sat_var, 0.0, 3.0, 3)
add_slider("Value (Brightness) Scale", val_var, 0.0, 3.0, 4)

# Initial display
update_image()

root.mainloop()
