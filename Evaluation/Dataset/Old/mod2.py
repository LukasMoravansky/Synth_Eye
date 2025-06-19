import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---- Load images ----
synthetic_bgr = cv2.imread(r"C:\projects\Synth_Eye\Evaluation\Dataset\Image_002.png")
real_bgr = cv2.imread(r"C:\projects\Data\Dataset_v1\images\test\Image_021.png")

synthetic_rgb = cv2.cvtColor(synthetic_bgr, cv2.COLOR_BGR2RGB)
real_rgb = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2RGB)

# ---- Color styling function ----
def adjust_color_style(image, alpha=1.0, beta=0, warm_r=1.0, warm_g=1.0, warm_b=1.0):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    warm_filter = np.array([[[warm_r, warm_g, warm_b]]])
    warmed = np.clip(adjusted * warm_filter, 0, 255).astype(np.uint8)
    return warmed

# ---- Setup figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(left=0.25, bottom=0.4)

# ---- Show initial images ----
adjusted_image = adjust_color_style(synthetic_rgb)
img_display = ax1.imshow(adjusted_image)
ax1.set_title("Adjusted Synthetic Image")
ax1.axis('off')

ax2.imshow(real_rgb)
ax2.set_title("Real-World Image")
ax2.axis('off')

# ---- Sliders ----
ax_alpha = plt.axes([0.25, 0.30, 0.65, 0.03])
ax_beta = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_r = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_g = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_b = plt.axes([0.25, 0.10, 0.65, 0.03])

slider_alpha = Slider(ax_alpha, 'Alpha (Contrast)', 0.5, 2.0, valinit=1.0)
slider_beta = Slider(ax_beta, 'Beta (Brightness)', -50, 50, valinit=0)
slider_r = Slider(ax_r, 'Warm R', 0.8, 1.2, valinit=1.0)
slider_g = Slider(ax_g, 'Warm G', 0.8, 1.2, valinit=1.0)
slider_b = Slider(ax_b, 'Warm B', 0.8, 1.2, valinit=0.98)

# ---- Update function ----
def update(val):
    alpha = slider_alpha.val
    beta = slider_beta.val
    warm_r = slider_r.val
    warm_g = slider_g.val
    warm_b = slider_b.val

    styled = adjust_color_style(synthetic_rgb, alpha, beta, warm_r, warm_g, warm_b)
    img_display.set_data(styled)
    fig.canvas.draw_idle()

# ---- Connect sliders ----
slider_alpha.on_changed(update)
slider_beta.on_changed(update)
slider_r.on_changed(update)
slider_g.on_changed(update)
slider_b.on_changed(update)

plt.show()
