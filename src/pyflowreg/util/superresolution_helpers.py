"""
Super-resolution helpers for warping images at an upscaled resolution.

Provides utilities that upsample an image together with its displacement
field and apply the warp at the higher resolution.
"""

import cv2
import numpy as np


def warp_image_highres(image, flow, scale=4):
    """
    Warp an image with a displacement field at an upscaled resolution.

    Upsamples ``image`` by ``scale`` with bicubic interpolation and
    ``flow`` with bilinear interpolation (displacements multiplied by
    ``scale``), then backward-warps the upsampled image with
    ``cv2.remap``: the output pixel at ``(y, x)`` is sampled bilinearly
    from the upsampled image at ``(x + scale*u, y + scale*v)``.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, C).
    flow : np.ndarray
        Displacement field of shape (H, W, 2), where ``flow[..., 0]`` is
        the horizontal (x) displacement and ``flow[..., 1]`` is the
        vertical (y) displacement, in pixels of the original resolution.
        Must be float32 (``cv2.remap`` coordinate-map requirement).
    scale : int, optional
        Upsampling factor applied to both image and flow. Default is 4.

    Returns
    -------
    np.ndarray
        Warped image of shape (H*scale, W*scale) or (H*scale, W*scale, C).
        Out-of-bounds samples use OpenCV's default constant (zero) border.
    """
    h, w = image.shape[:2]

    high_res_image = cv2.resize(
        image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
    )

    new_flow = cv2.resize(flow, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    new_flow[:, :, 0] *= scale
    new_flow[:, :, 1] *= scale

    new_flow[:, :, 0] += np.arange(w * scale)
    new_flow[:, :, 1] += np.arange(h * scale)[:, np.newaxis]

    remapped_image = cv2.remap(high_res_image, new_flow, None, cv2.INTER_LINEAR)

    return remapped_image
