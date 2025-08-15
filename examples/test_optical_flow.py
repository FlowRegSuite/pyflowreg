import numpy as np
import pyflowreg as pfr
import cv2

# Create simple test images with known displacement
img_size = 100
shift_x = 5
shift_y = 3

# Create a simple pattern
x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))
frame1 = np.sin(x * 0.1) * np.cos(y * 0.1)

# Create second frame with known shift
frame2 = np.zeros_like(frame1)
frame2[shift_y:, shift_x:] = frame1[:-shift_y, :-shift_x]

# Normalize
frame1 = (frame1 - frame1.min()) / (frame1.max() - frame1.min())
frame2 = (frame2 - frame2.min()) / (frame2.max() - frame2.min())

# Add channel dimension
frame1 = frame1[:, :, np.newaxis]
frame2 = frame2[:, :, np.newaxis]

print(f"Testing optical flow with known displacement: dx={shift_x}, dy={shift_y}")

# Compute optical flow
w = pfr.get_displacement(frame1, frame2, alpha=(2, 2), levels=20, iterations=50)

# Extract displacements
u = w[:, :, 0]  # y-displacement
v = w[:, :, 1]  # x-displacement

# Check central region (avoid boundaries)
central_slice = slice(20, 80)
u_central = u[central_slice, central_slice]
v_central = v[central_slice, central_slice]

mean_u = np.mean(u_central)
mean_v = np.mean(v_central)

print(f"Expected displacement: u={shift_y}, v={shift_x}")
print(f"Computed displacement (mean): u={mean_u:.2f}, v={mean_v:.2f}")
print(f"Error: u_error={abs(mean_u - shift_y):.2f}, v_error={abs(mean_v - shift_x):.2f}")

# Test warping
from pyflowreg.optical_flow import imregister_wrapper
warped = imregister_wrapper(frame2, u, v, frame1)

# Compute error
diff = np.abs(warped - frame1[1:-1, 1:-1])
print(f"Mean absolute error after warping: {np.mean(diff):.4f}")