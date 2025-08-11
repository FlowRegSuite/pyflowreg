import numpy as np, cv2
from pyflowreg.optical_flow import get_displacement

# --- synthetic pair ---------------------------------------------------------
H = W = 128
base = np.zeros((H, W), np.float64)
cv2.rectangle(base, (32, 32), (96, 96), 1.0, -1)

dy = 2                                # true vertical shift
shifted = np.zeros_like(base)
shifted[dy:, :] = base[:-dy, :]

im1 = cv2.GaussianBlur(base,    (0, 0), 2)
im2 = cv2.GaussianBlur(shifted, (0, 0), 2)

# two identical channels so cv2.resize keeps the 3‑D shape
im1 = np.stack([im1, im1], axis=2)
im2 = np.stack([im2, im2], axis=2)

# --- optical flow -----------------------------------------------------------
w = get_displacement(im1, im2, alpha=(2, 2), levels=0,
        iterations=50, a_data=0.45, a_smooth=1)

u, v = w[..., 0], w[..., 1]
print("Expected: median u = 0.000, median v = 7.000")
print(f"Observed: median u = {np.median(u):.3f}, median v = {np.median(v):.3f}")
print(f"Observed: max|u| = {np.max(np.abs(u)):.3f}, max|v| = {np.max(np.abs(v)):.3f}")

# --- visualisation ----------------------------------------------------------
ang = np.arctan2(v, u) + np.pi          # 0‥2π
mag = np.sqrt(u*u + v*v)
hsv = np.zeros((H, W, 3), np.uint8)
hsv[..., 0] = (ang / (2*np.pi) * 179).astype(np.uint8)
hsv[..., 1] = 255
hsv[..., 2] = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('im1', (im1[..., 0] * 255).astype(np.uint8))
cv2.imshow('im2', (im2[..., 0] * 255).astype(np.uint8))
cv2.imshow('flow (HSV)', flow_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
