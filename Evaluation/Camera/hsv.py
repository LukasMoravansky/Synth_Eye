import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Load image
def load_image():
    path = 'Test.png'
    if path:
        global img_original
        img_original = cv2.imread(path)
        update_image()

# Update image based on slider values
def update_image(*args):
    if img_original is None:
        return

    hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)

    lower = np.array([
        h_min.get(),
        s_min.get(),
        v_min.get()
    ])
    upper = np.array([
        h_max.get(),
        s_max.get(),
        v_max.get()
    ])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img_original, img_original, mask=mask)

    cv2.imshow("Filtered Image (No Blur)", result)
    cv2.waitKey(1)

# GUI
root = tk.Tk()
root.title("HSV Filter Tool (No Blur)")

img_original = None

# Load image button
tk.Button(root, text="Load Image", command=load_image).pack()

# HSV sliders
h_min = tk.Scale(root, from_=0, to=179, orient="horizontal", label="H Min", command=update_image)
h_min.pack()
h_max = tk.Scale(root, from_=0, to=179, orient="horizontal", label="H Max", command=update_image)
h_max.set(179)
h_max.pack()

s_min = tk.Scale(root, from_=0, to=255, orient="horizontal", label="S Min", command=update_image)
s_min.pack()
s_max = tk.Scale(root, from_=0, to=255, orient="horizontal", label="S Max", command=update_image)
s_max.set(255)
s_max.pack()

v_min = tk.Scale(root, from_=0, to=255, orient="horizontal", label="V Min", command=update_image)
v_min.pack()
v_max = tk.Scale(root, from_=0, to=255, orient="horizontal", label="V Max", command=update_image)
v_max.set(255)
v_max.pack()

# Run GUI loop
root.mainloop()
cv2.destroyAllWindows()
