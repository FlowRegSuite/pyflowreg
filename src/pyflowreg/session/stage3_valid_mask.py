"""
Stage 3: Valid mask alignment and final session mask computation.

Aligns per-sequence valid masks to the common reference frame and
computes the final session-wide valid mask as the intersection of all masks.

Mirrors MATLAB get_session_valid_index_v3_progressprint.m logic.
"""

from pathlib import Path
from time import time
from typing import List, Optional

import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from pyflowreg.core.warping import imregister_binary, align_sequence
from pyflowreg.session.config import SessionConfig
from pyflowreg.session.stage1_compensate import (
    discover_input_files,
    load_or_create_status,
    save_status,
)


def load_idx_and_compute_mask(idx_path: Path) -> np.ndarray:
    """
    Load per-frame valid indices and compute temporal AND mask.

    Parameters
    ----------
    idx_path : Path
        Path to idx.hdf file

    Returns
    -------
    vmask : ndarray, shape (H, W), dtype=bool
        Valid mask (True where all frames are valid)

    Notes
    -----
    Mirrors MATLAB logic (get_session_valid_index_v3.m lines 34-38):
        r = get_video_file_reader(fullfile(seq_dir,'idx.hdf'));
        vmask = r.read_frames(1:r.frame_count);
        vmask = vmask > 0;
        vmask = all(vmask, 4);  # AND across time
    """
    from pyflowreg.util.io.factory import get_video_file_reader

    reader = get_video_file_reader(str(idx_path))
    # Use array indexing instead of read_frames (which doesn't exist)
    vmask = reader[:]  # Read all frames

    # Convert to boolean
    vmask = vmask > 0

    # AND across time (axis 0 for THWC format)
    # Handle both 3D (T,H,W) and 4D (T,H,W,C) formats
    if vmask.ndim == 4:
        vmask = vmask[:, :, :, 0]  # Take first channel if multi-channel

    vmask = np.all(vmask, axis=0)

    return vmask.astype(bool)


def save_mask_png(mask: np.ndarray, path: Path):
    """
    Save boolean mask as PNG (0 or 255).

    Parameters
    ----------
    mask : ndarray, shape (H, W), dtype=bool
        Boolean mask
    path : Path
        Output path for PNG
    """
    if PIL_AVAILABLE:
        # Use PIL for better compatibility
        img = Image.fromarray((mask.astype(np.uint8) * 255))
        img.save(str(path))
    else:
        # Fallback to numpy/opencv if needed
        try:
            import cv2

            cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        except ImportError:
            # Last resort: just save as numpy
            np.save(str(path.with_suffix(".npy")), mask)
            print("Warning: PIL not available, saved as .npy instead of .png")


def align_sequence_video(
    compensated_path: Path,
    output_path: Path,
    displacement: np.ndarray,
    reference_average: np.ndarray,
    *,
    resume: bool = True,
    chunk_size: int = 64,
    output_format: str = "TIFF",
) -> Path:
    """
    Warp a compensated recording into the session reference frame and persist it.

    Parameters
    ----------
    compensated_path : Path
        Path to the per-recording compensated video.
    output_path : Path
        Destination path for the aligned video.
    displacement : ndarray, shape (H, W, 2)
        Displacement field that maps the recording into the reference frame.
    reference_average : ndarray
        Reference image used to fill out-of-bounds regions.
    resume : bool, default=True
        Skip processing when the aligned video already exists.
    chunk_size : int, default=64
        Number of frames processed per batch.
    output_format : str, default="TIFF"
        Output format for the aligned video.

    Returns
    -------
    Path
        Path to the aligned video.
    """
    if resume and output_path.exists():
        print(f"  Skipping existing {output_path.name}")
        return output_path

    print(f"  Creating {output_path.name}...")
    start_time = time()

    from pyflowreg.util.io.factory import (
        get_video_file_reader,
        get_video_file_writer,
    )

    reference = reference_average.astype(np.float64, copy=False)
    reader = get_video_file_reader(str(compensated_path))
    writer = get_video_file_writer(str(output_path), output_format=output_format)

    total_frames = len(reader)
    frame_dtype = reader.dtype

    for start_idx in range(0, total_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, total_frames)
        batch = reader[slice(start_idx, end_idx)]

        if batch.ndim == 3:  # (H, W, C) single frame due to binning logic
            batch = batch[np.newaxis, ...]

        # Use align_sequence for batch processing
        aligned_batch = align_sequence(
            batch,
            displacement,
            reference,
            interpolation_method="cubic",
            n_workers=1,  # Use single worker since we're already batching
        )

        # Preserve dtype
        aligned_batch = aligned_batch.astype(frame_dtype, copy=False)
        writer.write_frames(aligned_batch)

        # Progress indication
        if (end_idx % 500 == 0) or (end_idx == total_frames):
            progress = (end_idx / total_frames) * 100
            print(f"    Progress: {progress:.1f}% ({end_idx}/{total_frames} frames)")

    reader.close()
    writer.close()

    elapsed = time() - start_time
    print(f"  Saved aligned video to {output_path.name} after {elapsed:.2f} seconds.")
    return output_path


def save_final_results(
    final_results: Path,
    final_valid: np.ndarray,
    aligned_valid_masks: List[np.ndarray],
    per_seq_valid_masks: List[np.ndarray],
    displacement_fields: List[np.ndarray],
    temporal_averages: List[np.ndarray],
    compensated_h5_paths: List[Path],
    reference_average: np.ndarray,
    middle_idx: int,
    aligned_video_paths: List[Path] = None,
):
    """
    Save all final results including masks and metadata.

    Parameters
    ----------
    final_results : Path
        Output directory for final results
    final_valid : ndarray
        Final session-wide valid mask
    aligned_valid_masks : list[ndarray]
        Per-sequence masks aligned to reference
    per_seq_valid_masks : list[ndarray]
        Per-sequence masks in original coordinates
    displacement_fields : list[ndarray]
        Displacement fields to reference
    temporal_averages : list[ndarray]
        Temporal averages for each sequence
    compensated_h5_paths : list[Path]
        Paths to compensated.hdf5 files
    reference_average : ndarray
        Reference temporal average
    middle_idx : int
        Index of center reference
    aligned_video_paths : list[Path], optional
        Paths to aligned video files

    Notes
    -----
    Mirrors MATLAB output (get_session_valid_index_v3.m lines 102-107):
        save(..., 'final_valid', 'aligned_valid_masks', 'per_seq_valid_masks',
             'displacement_fields', 'temporal_averages', 'compensated_h5_paths',
             'reference_average', 'middle_idx');
    """
    final_results.mkdir(parents=True, exist_ok=True)

    # Save final mask (PNG)
    save_mask_png(final_valid, final_results / "final_valid_idx.png")

    # Save comprehensive .npz with all data
    npz_path = final_results / "final_valid_idx.npz"
    save_dict = {
        "final_valid": final_valid,
        "aligned_valid_masks": np.array(aligned_valid_masks),
        "per_seq_valid_masks": np.array(per_seq_valid_masks),
        "displacement_fields_u": np.array([w[:, :, 0] for w in displacement_fields]),
        "displacement_fields_v": np.array([w[:, :, 1] for w in displacement_fields]),
        "temporal_averages": np.array(temporal_averages),
        "compensated_h5_paths": np.array([str(p) for p in compensated_h5_paths]),
        "reference_average": reference_average,
        "middle_idx": middle_idx,
    }

    # Add aligned video paths if provided
    if aligned_video_paths is not None:
        save_dict["aligned_video_paths"] = np.array(
            [str(p) for p in aligned_video_paths]
        )

    np.savez(str(npz_path), **save_dict)

    # Also save .mat for MATLAB compatibility
    try:
        from scipy.io import savemat

        mat_path = final_results / "final_valid_idx.mat"

        # Prepare displacement fields as cell array equivalent
        # (save u and v separately for MATLAB)
        savemat(
            str(mat_path),
            {
                "final_valid": final_valid,
                "aligned_valid_masks": np.array(aligned_valid_masks),
                "per_seq_valid_masks": np.array(per_seq_valid_masks),
                "displacement_fields_u": np.array(
                    [w[:, :, 0] for w in displacement_fields]
                ),
                "displacement_fields_v": np.array(
                    [w[:, :, 1] for w in displacement_fields]
                ),
                "temporal_averages": np.array(temporal_averages),
                "compensated_h5_paths": [str(p) for p in compensated_h5_paths],
                "reference_average": reference_average,
                "middle_idx": middle_idx + 1,  # MATLAB is 1-indexed
            },
        )
    except ImportError:
        print("Warning: scipy not available, skipping .mat file creation")

    print(f"Saved final results to {final_results}")


def run_stage3(
    config: SessionConfig,
    middle_idx: Optional[int] = None,
    displacement_fields: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """
    Run Stage 3: Align valid masks and compute final session mask.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    middle_idx : int, optional
        Index of center reference (if None, will load from Stage 2)
    displacement_fields : list[ndarray], optional
        Displacement fields from Stage 2 (if None, will load)

    Returns
    -------
    final_valid : ndarray, shape (H, W), dtype=bool
        Final session-wide valid mask (AND of all aligned masks)

    Notes
    -----
    Mirrors MATLAB get_session_valid_index_v3_progressprint.m logic:

    Step 1 (lines 24-53): Load idx.hdf, compute per-sequence masks
    Step 2 (lines 55-91): Compute displacements and align masks
    Step 3 (lines 95-109): Compute final mask as AND of aligned masks

    For each sequence:
    1. Load idx.hdf and compute temporal AND
    2. Warp mask to reference using w_to_reference
    3. Compose with in-bounds mask (idx_warp logic)
    4. Save aligned mask

    Final mask = AND of all aligned masks

    Examples
    --------
    >>> config = SessionConfig.from_toml("session.toml")
    >>> final_mask = run_stage3(config)
    """
    start_time = time()

    # Setup paths
    output_root, final_results = config.resolve_output_paths()
    final_results.mkdir(parents=True, exist_ok=True)

    # Discover inputs
    input_files = discover_input_files(config)
    num_records = len(input_files)

    # Check if already complete
    final_valid_path = final_results / "final_valid_idx.png"
    if config.resume and final_valid_path.exists():
        status_path = final_results / "status.json"
        if status_path.exists():
            status = load_or_create_status(final_results)
            if status.get("stage3") == "done":
                print("Stage 3 already complete, loading final mask...")
                # Load from npz if available
                npz_path = final_results / "final_valid_idx.npz"
                if npz_path.exists():
                    data = np.load(str(npz_path))
                    return data["final_valid"]

    print("Starting step 1, loading valid indices and temporal averages...\n")

    # Step 1: Load per-frame indices and compute per-sequence masks
    per_seq_valid_masks = []
    compensated_h5_paths = []
    temporal_averages = []

    for i, input_file in enumerate(input_files):
        seq_dir = output_root / input_file.stem
        print(f"Processing folder {seq_dir.name}...")
        processing_timer = time()

        # Load and compute per-sequence valid mask
        idx_path = seq_dir / "idx.hdf"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Missing idx.hdf for {input_file.stem}. Run Stage 1 with save_valid_idx=True"
            )

        vmask = load_idx_and_compute_mask(idx_path)
        per_seq_valid_masks.append(vmask)

        # Save per-sequence mask
        valid_png_path = final_results / f"{input_file.stem}_valid_idx.png"
        save_mask_png(vmask, valid_png_path)

        # Load compensated file path and temporal average
        # Handle both .hdf5 and .HDF5
        h5_candidates = [seq_dir / "compensated.hdf5", seq_dir / "compensated.HDF5"]
        h5_path = next((p for p in h5_candidates if p.exists()), h5_candidates[0])
        compensated_h5_paths.append(h5_path)

        avg_path = seq_dir / "temporal_average.npy"
        if avg_path.exists():
            temporal_averages.append(np.load(str(avg_path)))
        else:
            # If not cached, compute it
            from pyflowreg.util.io.factory import get_video_file_reader

            vr = get_video_file_reader(str(h5_path))
            # Use array indexing instead of read_frames (which doesn't exist)
            vid = vr[:]  # Read all frames
            temporal_averages.append(np.mean(vid, axis=0))

        processing_toc = time() - processing_timer
        print(f"Done after {processing_toc:.2f} seconds.")
        remaining = num_records - (i + 1)
        if remaining > 0:
            print(f"{remaining} folder(s) remaining.")

    # Step 2: Load/compute displacements and align masks
    align_tic = time()

    if middle_idx is None:
        # Resolve center from config
        middle_idx, _ = config.resolve_center_file(input_files)

    reference_average = temporal_averages[middle_idx]

    # Load displacement fields if not provided
    if displacement_fields is None:
        displacement_fields = []
        for i, input_file in enumerate(input_files):
            seq_dir = output_root / input_file.stem
            w_path = seq_dir / "w_to_reference.npz"

            if not w_path.exists():
                raise FileNotFoundError(
                    f"Missing w_to_reference.npz for {input_file.stem}. Run Stage 2 first"
                )

            w_data = np.load(str(w_path))
            w = np.stack([w_data["u"], w_data["v"]], axis=-1)
            displacement_fields.append(w)

    print(
        f"\nStep 1 done.\n\nStarting step 2, computation of displacement fields "
        f"and alignment of valid masks with respect to folder {input_files[middle_idx].stem}...\n"
    )

    # Align each mask to reference
    aligned_valid_masks = []

    for i in range(num_records):
        w = displacement_fields[i]
        m = per_seq_valid_masks[i]

        # Warp mask using nearest interpolation + in-bounds composition
        aligned_m = imregister_binary(m, w[:, :, 0], w[:, :, 1])
        aligned_valid_masks.append(aligned_m)

        # Save aligned mask
        aligned_valid_png_path = (
            final_results / f"{input_files[i].stem}_valid_idx_aligned.png"
        )
        save_mask_png(aligned_m, aligned_valid_png_path)

        remaining = num_records - (i + 1)
        if remaining > 0:
            print(f"{remaining} folder(s) remaining.")

    align_toc = time() - align_tic
    print(f"Done after {align_toc:.2f} seconds.\n")

    # Step 2b: Align video frames and persist (optional)
    aligned_video_paths = []
    print("Aligning video frames to reference coordinate system...\n")

    for i in range(num_records):
        print(f"Aligning video {i+1}/{num_records}: {input_files[i].stem}")

        # Determine output path based on config
        ext = (
            "tif"
            if config.align_output_format == "TIFF"
            else config.align_output_format.lower()
        )
        aligned_video_path = final_results / f"aligned_{input_files[i].stem}.{ext}"

        # Align and save the video
        aligned_path = align_sequence_video(
            compensated_h5_paths[i],
            aligned_video_path,
            displacement_fields[i],
            reference_average,
            resume=config.resume,
            chunk_size=config.align_chunk_size,
            output_format=config.align_output_format,
        )
        aligned_video_paths.append(aligned_path)

        remaining = num_records - (i + 1)
        if remaining > 0:
            print(f"{remaining} video(s) remaining.\n")

    # Step 3: Compute final valid mask
    print("\nStarting step 3, computing final valid index mask...\n")
    final_timer = time()

    # Initialize as all True, then AND with each aligned mask
    final_valid = np.ones(
        (reference_average.shape[0], reference_average.shape[1]), dtype=bool
    )

    for aligned_mask in aligned_valid_masks:
        final_valid = final_valid & aligned_mask

    # Save all results
    save_final_results(
        final_results,
        final_valid,
        aligned_valid_masks,
        per_seq_valid_masks,
        displacement_fields,
        temporal_averages,
        compensated_h5_paths,
        reference_average,
        middle_idx,
        aligned_video_paths,
    )

    # Mark stage 3 as complete
    status = load_or_create_status(final_results)
    status["stage3"] = "done"
    save_status(final_results, status)

    final_toc = time() - final_timer
    print(f"Done after {final_toc:.2f} seconds.")

    total_time = time() - start_time
    print(f"\nThe complete job took {total_time:.2f} seconds.\n")

    return final_valid
