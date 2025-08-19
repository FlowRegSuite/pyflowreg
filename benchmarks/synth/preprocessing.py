import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def normalize(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    rng = mx - mn
    if rng < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (img - mn) / rng


def imgaussfilt(img, sigma):
    return gaussian_filter(img, sigma, mode='constant')


class DefaultProcessor:
    def __init__(self, sigma=None):
        self.sigma = sigma if sigma is not None else 0

    def process(self, frame):
        if self.sigma == 0:
            return normalize(frame.astype(np.float32))
        return normalize(imgaussfilt(frame, self.sigma).astype(np.float32))


class GradientProcessor:
    def __init__(self, sigma=1.5):
        self.sigma = sigma

    def process(self, frame):
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, -1)

        result = []
        for ch in range(frame.shape[-1]):
            # central differences:
            f = normalize(imgaussfilt(frame[..., ch], self.sigma))
            
            f = cv2.copyMakeBorder(f, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
            fx = f[1:-1, 2:] - f[1:-1, 0:-2]
            fx *= 0.5
            fy = f[2:, 1:-1] - f[0:-2, 1:-1]
            fy *= 0.5
            result.append(np.stack([fx, fy], -1))

        return np.concatenate(result, axis=-1).astype(np.float32)