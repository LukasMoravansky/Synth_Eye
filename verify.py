import numpy as np
import matplotlib
import pandas as pd
import scipy
import cv2
import pypylon
import ultralytics

print('[INFO] Installed package versions:') 
print(f'[INFO] - NumPy: {np.__version__}') 
print(f'[INFO] - Matplotlib: {matplotlib.__version__}') 
print(f'[INFO] - Pandas: {pd.__version__}') 
print(f'[INFO] - SciPy: {scipy.__version__}') 
print(f'[INFO] - OpenCV: {cv2.__version__}')
print(f'[INFO] - pypylon: {pypylon.__version__ if hasattr(pypylon, "__version__") else "Version info not available."}') 
print(f'[INFO] - ultralytics: {ultralytics.__version__ if hasattr(ultralytics, "__version__") else "Version info not available."}')
print('[INFO] Test passed successfully!')
