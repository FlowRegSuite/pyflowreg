import numpy as np
from skimage.registration import phase_cross_correlation

from pyflowreg.util.resize_util import resize_image_cv2


def estimate_rigid_xcorr_2d(
    ref_img,
    mov_img,
    target_hw=(256, 256),
    up=1,
    normalization="phase",
    disambiguate=True,
    weight=None,
):
    """
    Estimate rigid displacement between 2D images using phase cross-correlation.

    Parameters
    ----------
    ref_img : np.ndarray
        Reference image, shape (H, W) or (H, W, C)
    mov_img : np.ndarray
        Moving image, shape (H, W) or (H, W, C)
    target_hw : tuple or int
        Target size for downsampling before correlation
    up : int
        Upsampling factor for subpixel accuracy (1 = no upsampling)
    normalization : str
        Normalization method for phase_cross_correlation
    disambiguate : bool
        Whether to disambiguate sign of shift
    weight : np.ndarray or None
        Channel weights for multi-channel images

    Returns
    -------
    np.ndarray
        Displacement vector [dx, dy] (negated for backward warp convention)
    """
    # Handle multi-channel images
    if ref_img.ndim == 3 and ref_img.shape[2] > 1:
        if weight is not None:
            w = weight.reshape(-1).astype(np.float32)
            w /= w.sum()
            ref_img = np.tensordot(ref_img, w, axes=([2], [0]))
            mov_img = np.tensordot(mov_img, w, axes=([2], [0]))
        else:
            ref_img = ref_img.mean(axis=2)
            mov_img = mov_img.mean(axis=2)
    elif ref_img.ndim == 3:
        ref_img = ref_img[..., 0]
        mov_img = mov_img[..., 0]

    # Get current dimensions
    H, W = ref_img.shape

    # Parse target size
    if isinstance(target_hw, (int, float)):
        target_hw = (int(target_hw), int(target_hw))

    Th = H if target_hw is None else min(H, int(target_hw[0]))
    Tw = W if target_hw is None else min(W, int(target_hw[1]))

    # Validate upsampling factor
    if up < 1:
        print(
            f"Warning: upsampling factor {up} < 1 is invalid. "
            f"Setting to 1 (no upsampling)."
        )
        up = 1

    # Calculate scaling factors
    sy = H / Th
    sx = W / Tw

    # Downsample if needed
    ref_small = ref_img
    mov_small = mov_img
    if (Th, Tw) != (H, W):
        ref_small = resize_image_cv2(ref_img, (Tw, Th))
        mov_small = resize_image_cv2(mov_img, (Tw, Th))

    # Pre-whiten + Hann window to avoid integer-bin locking on periodic patterns
    ref_small = ref_small.astype(np.float32, copy=False)
    mov_small = mov_small.astype(np.float32, copy=False)
    ref_small = ref_small - ref_small.mean()
    mov_small = mov_small - mov_small.mean()
    hy = np.hanning(ref_small.shape[0]).astype(np.float32)
    hx = np.hanning(ref_small.shape[1]).astype(np.float32)
    win = hy[:, None] * hx[None, :]
    ref_small = ref_small * win
    mov_small = mov_small * win

    # Compute phase cross-correlation (returns (row, col) = (y, x) shift to apply to mov to align to ref)
    shift_rc, _, _ = phase_cross_correlation(
        ref_small,
        mov_small,
        upsample_factor=up,
        normalization=normalization,
        disambiguate=disambiguate,
    )

    ty, tx = (
        float(shift_rc[0]),
        float(shift_rc[1]),
    )  # translation for mov -> ref, in resized pixels

    # Scale back to original grid (resized->full)
    # We want the FORWARD shift [dx, dy]; PCC gives (ty, tx) that aligns mov to ref,
    # so forward is the negative of that, with scale correction:
    dx = -tx * sx
    dy = -ty * sy

    return np.array([dx, dy], dtype=np.float32)


if __name__ == "__main__":
    from scipy.ndimage import shift as ndi_shift
    from pyflowreg.core.warping import imregister_wrapper
    import matplotlib.pyplot as plt

    print("Testing 2D rigid cross-correlation alignment...")

    # Create reference image with a simple rectangle (like original example)
    ref = np.zeros((256, 256), dtype=np.float32)
    ref[100:150, 100:150] = 1.0
    ref += np.random.randn(*ref.shape) * 0.1  # Add noise

    # True displacement to apply to mov to align with ref
    true_dx_dy = np.array([15.0, -10.0], dtype=np.float32)

    # Create moved image by shifting ref with POSITIVE shifts (like 3D version)
    # ndi_shift uses (dy, dx) order
    mov = ndi_shift(ref, shift=(true_dx_dy[1], true_dx_dy[0]), order=1, mode="nearest")
    mov += np.random.randn(*mov.shape) * 0.1  # Add noise

    # Estimate displacement
    est = estimate_rigid_xcorr_2d(ref, mov, target_hw=128, up=10)
    error = np.abs(est - true_dx_dy)

    print(f"True shift [dx, dy]: {true_dx_dy}")
    print(f"Estimated  [dx, dy]: {est}")
    print(f"Error:               {error}")
    print(f"Max error:           {error.max():.3f} pixels")

    # Apply alignment using imregister_wrapper
    aligned = imregister_wrapper(
        mov, est[0], est[1], ref, interpolation_method="linear"
    )
    if aligned.ndim == 3:  # Remove channel dimension if added
        aligned = aligned[..., 0]

    alignment_error = np.mean(np.abs(aligned - ref))
    print(f"Alignment error:     {alignment_error:.6f}")

    # Visualize
    try:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(ref, cmap="gray")
        axes[0].set_title("Reference")
        axes[1].imshow(mov, cmap="gray")
        axes[1].set_title("Moving (shifted)")
        axes[2].imshow(aligned, cmap="gray")
        axes[2].set_title("Aligned")
        axes[3].imshow(np.abs(aligned - ref), cmap="hot", vmin=0, vmax=0.5)
        axes[3].set_title("Difference")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")

    assert error.max() < 1.0, f"Error too large: {error}"
    print("Test passed!")
