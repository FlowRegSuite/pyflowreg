import cv2
import numpy as np


def warp_image_highres(image, flow, scale=4):
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
