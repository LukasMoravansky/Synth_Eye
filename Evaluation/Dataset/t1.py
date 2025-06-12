import cv2
import numpy as np
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

# Load images (OpenCV BGR)
source_img = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\test\Image_Background_001.png")
target_img = cv2.imread(r"C:\projects\Synth_Eye\Data\Dataset_v1\images\train\Image_Background_001.png")

# Convert to RGB for skimage
source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

# Perform histogram matching
matched_rgb = match_histograms(target_rgb, source_rgb, channel_axis=-1)

# Convert back to BGR for OpenCV saving/displaying
matched_bgr = cv2.cvtColor((matched_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Save or show
#cv2.imwrite('matched_image.jpg', matched_bgr)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Source Image')
plt.imshow(source_rgb)
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Target Image')
plt.imshow(target_rgb)
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Histogram Matched')
plt.imshow(matched_rgb)
plt.axis('off')
plt.show()
