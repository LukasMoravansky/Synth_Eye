import torch
print(torch.cuda.is_available())         # ✅ True
print(torch.cuda.device_count())         # ✅ > 0
print(torch.cuda.get_device_name(0))     # ✅ NVIDIA RTX A4000

"""
if torch.cuda.is_available():
    print("NVIDIA GPU is available.")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("NVIDIA GPU is NOT available.")
"""