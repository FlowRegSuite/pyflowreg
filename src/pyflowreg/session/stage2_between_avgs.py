"""
Stage 2: Inter-sequence displacement computation.

Computes displacement fields between temporal averages to align
all recordings to a common reference frame.

Mirrors MATLAB align_full_v3_checkpoint.m and get_session_valid_index_v3 Stage 2 logic.
"""

from pathlib import Path
from time import time
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from pyflowreg.core.backend_registry import get_backend
from pyflowreg.core.warping import imregister_wrapper
from pyflowreg.session.config import SessionConfig
from pyflowreg.session.stage1_compensate import (
    atomic_save_npz,
    discover_input_files,
    load_or_create_status,
    save_status,
)
from pyflowreg.util.xcorr_prealignment import estimate_rigid_xcorr_2d


def mat2gray_ref(img: np.ndarray, ref: np.ndarray = None) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Parameters
    ----------
    img : ndarray
        Input image
    ref : ndarray, optional
        Reference image for min/max calculation (default: img)

    Returns
    -------
    ndarray
        Normalized image in [0, 1]
    """

    if ref is None:
        img_min = img.min()
        img_max = img.max()
    else:
        img_min = ref.min()
        img_max = ref.max()

    return (img - img_min) / (img_max - img_min)


def compute_between_displacement(
    reference_avg: np.ndarray,
    current_avg: np.ndarray,
    config: SessionConfig,
) -> np.ndarray:
    """
    Compute displacement field from current average to reference.

    Parameters
    ----------
    reference_avg : ndarray
        Reference temporal average
    current_avg : ndarray
        Current sequence temporal average
    config : SessionConfig
        Session configuration

    Returns
    -------
    w : ndarray, shape (H, W, 2)
        Displacement field [u, v]

    Notes
    -----
    Mirrors MATLAB logic (align_full_v3_checkpoint.m lines 103-107):
        w_init = get_displacement_cc(reference_average, current_avg);
        sigma = 6;
        img1 = mat2gray(imgaussfilt(reference_average, sigma));
        img2 = mat2gray(imgaussfilt(current_avg, sigma));
        w = get_displacement(img1, img2, 'alpha', 25, 'iterations', 100, 'uv', w_init(:,:,1), w_init(:,:,2));
    """

    # Smooth both images
    sigma = config.sigma_smooth
    ref_smooth = gaussian_filter(reference_avg, sigma=sigma)
    cur_smooth = gaussian_filter(current_avg, sigma=sigma)

    img1 = mat2gray_ref(ref_smooth)
    img2 = mat2gray_ref(cur_smooth, ref_smooth)

    # Rigid initialization via cross-correlation
    # Use existing utility function with proper backward warp convention
    shift_2d = estimate_rigid_xcorr_2d(img1, img2, up=config.cc_upsample)

    # Create constant displacement field from 2D shift
    dx, dy = shift_2d[0], shift_2d[1]
    H, W = img2.shape[:2]
    w_init = np.zeros((H, W, 2), dtype=np.float32)
    w_init[:, :, 0] = dx
    w_init[:, :, 1] = dy

    print(f"  Rigid pre-alignment shift: dx={dx:.2f}, dy={dy:.2f}")

    img2_dims = img2.shape
    img2 = imregister_wrapper(
        img2,
        w_init[:, :, 0],
        w_init[:, :, 1],
        img1,
        interpolation_method="cubic",
    )
    img2 = img2.reshape(img2_dims)

    # Get displacement function based on configured backend
    backend_factory = get_backend(config.flow_backend)
    get_displacement_func = backend_factory(**config.backend_params)

    # Refine with optical flow
    # Convert scalar alpha to tuple (alpha_x, alpha_y) if needed
    alpha = config.alpha_between
    if not isinstance(alpha, (tuple, list)):
        alpha = (alpha, alpha)

    w = get_displacement_func(
        img1,
        img2,
        alpha=alpha,
        iterations=config.iterations_between,
    )

    return w + w_init


def load_temporal_averages(
    output_root: Path, input_files: List[Path]
) -> List[np.ndarray]:
    """
    Load temporal averages from Stage 1 outputs.

    Parameters
    ----------
    output_root : Path
        Root directory containing Stage 1 outputs
    input_files : list[Path]
        List of input files (to derive output folder names)

    Returns
    -------
    list[ndarray]
        Temporal averages for each recording

    Raises
    ------
    FileNotFoundError
        If any temporal_average.npy is missing
    """
    temporal_averages = []

    for input_file in input_files:
        output_folder = output_root / input_file.stem
        avg_path = output_folder / "temporal_average.npy"

        if not avg_path.exists():
            raise FileNotFoundError(
                f"Missing temporal average: {avg_path}. "
                f"Run Stage 1 first for {input_file.name}"
            )

        avg = np.load(str(avg_path))
        temporal_averages.append(avg)

    return temporal_averages


def run_stage2(config: SessionConfig) -> Tuple[int, Path, List[np.ndarray]]:
    """
    Run Stage 2: Compute inter-sequence displacements.

    Parameters
    ----------
    config : SessionConfig
        Session configuration

    Returns
    -------
    middle_idx : int
        Index of center reference recording
    center_file : Path
        Path to center reference file
    displacement_fields : list[ndarray]
        Displacement fields for each recording (zeros for center)

    Notes
    -----
    Mirrors MATLAB logic:
    - align_full_v3_checkpoint.m lines 82-113
    - get_session_valid_index_v3.m lines 55-91

    For the center recording, displacement is zeros.
    For others:
    1. Rigid init via cross-correlation (upsample=4)
    2. Gaussian smooth (sigma=6)
    3. Refine with OF (alpha=25, iterations=100)

    Results saved as w_to_reference.npz in each output folder.

    Examples
    --------
    >>> config = SessionConfig.from_toml("session.toml")
    >>> middle_idx, center_file, displacements = run_stage2(config)
    """
    start_time = time()

    # Setup paths
    output_root, final_results = config.resolve_output_paths()

    # Discover inputs
    input_files = discover_input_files(config)
    num_records = len(input_files)

    # Load temporal averages
    print("\nLoading temporal averages from Stage 1...")
    temporal_averages = load_temporal_averages(output_root, input_files)

    # Resolve center reference
    middle_idx, center_file = config.resolve_center_file(input_files)
    reference_average = temporal_averages[middle_idx]

    print(
        f"\nStep 1 motion correction done.\n\n"
        f"Starting step 2, computation of temporal averages for each sequence "
        f"and displacement estimation with respect to {center_file.name}...\n"
    )

    # Compute displacements
    displacement_fields = []

    for idx in range(num_records):
        output_folder = output_root / input_files[idx].stem
        w_path = output_folder / "w_to_reference.npz"

        # Check resume
        if config.resume and w_path.exists():
            print(f"Loading existing displacement for {input_files[idx].stem}")
            w_data = np.load(str(w_path))
            w = np.stack([w_data["u"], w_data["v"]], axis=-1)
            displacement_fields.append(w)
        else:
            current_avg = temporal_averages[idx]

            if idx == middle_idx:
                # Center: zero displacement
                w = np.zeros(
                    (current_avg.shape[0], current_avg.shape[1], 2), dtype=np.float32
                )
            else:
                # Non-center: compute displacement
                print(f"Computing displacement for {input_files[idx].stem}...")
                w = compute_between_displacement(reference_average, current_avg, config)

            # Save atomically (write-to-temp then replace)
            atomic_save_npz(w_path, u=w[:, :, 0], v=w[:, :, 1])
            displacement_fields.append(w)

            # Update status
            status = load_or_create_status(output_folder)
            status["stage2"] = "done"
            save_status(output_folder, status)

        # Progress
        remaining = num_records - (idx + 1)
        if remaining > 0:
            print(f"{remaining} file(s) remaining.")

    elapsed = time() - start_time
    print(f"\nDone after {elapsed:.2f} seconds.\n")

    return middle_idx, center_file, displacement_fields
