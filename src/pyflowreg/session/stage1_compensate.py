"""
Stage 1: Per-recording motion correction.

Discovers input recordings, compensates each sequence independently,
and computes temporal averages for inter-sequence alignment.

Mirrors MATLAB align_full_v3_checkpoint.m Stage 1 logic.
"""

import json
from pathlib import Path
from time import time
from typing import Dict, List, Optional

import numpy as np

from pyflowreg.motion_correction.compensate_recording import compensate_recording
from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.session.config import SessionConfig, get_array_task_id


def discover_input_files(config: SessionConfig) -> List[Path]:
    """
    Discover input files matching pattern in root directory.

    Parameters
    ----------
    config : SessionConfig
        Session configuration

    Returns
    -------
    list[Path]
        Sorted list of matching input files

    Raises
    ------
    ValueError
        If no files found matching pattern
    """
    input_files = sorted(config.root.glob(config.pattern))

    if not input_files:
        raise ValueError(
            f"No files found matching pattern '{config.pattern}' "
            f"in directory {config.root}"
        )

    return input_files


def load_or_create_status(output_folder: Path) -> Dict:
    """
    Load existing status.json or create empty status dict.

    Parameters
    ----------
    output_folder : Path
        Folder to check for status.json

    Returns
    -------
    dict
        Status dictionary with completion flags
    """
    status_path = output_folder / "status.json"

    if status_path.exists():
        with open(status_path, "r") as f:
            return json.load(f)
    else:
        return {}


def save_status(output_folder: Path, status: Dict):
    """
    Atomically save status.json.

    Parameters
    ----------
    output_folder : Path
        Folder to save status.json
    status : dict
        Status dictionary to save
    """
    status_path = output_folder / "status.json"
    temp_path = status_path.with_suffix(".json.tmp")

    # Write to temp file first
    with open(temp_path, "w") as f:
        json.dump(status, f, indent=2)

    # Atomic rename
    temp_path.replace(status_path)


def is_stage1_complete(output_folder: Path) -> bool:
    """
    Check if Stage 1 is complete for a sequence.

    Parameters
    ----------
    output_folder : Path
        Output folder for the sequence

    Returns
    -------
    bool
        True if compensated.hdf5, temporal_average.npy, and status indicate completion
    """
    # Handle both .hdf5 and .HDF5
    h5_candidates = [
        output_folder / "compensated.hdf5",
        output_folder / "compensated.HDF5",
    ]
    has_compensated = any(p.exists() for p in h5_candidates)

    temporal_avg_npy = output_folder / "temporal_average.npy"
    status = load_or_create_status(output_folder)

    return (
        has_compensated and temporal_avg_npy.exists() and status.get("stage1") == "done"
    )


def compute_and_save_temporal_average(
    compensated_path: Path, output_folder: Path
) -> np.ndarray:
    """
    Compute temporal average from compensated video and save.

    Parameters
    ----------
    compensated_path : Path
        Path to compensated.hdf5 file
    output_folder : Path
        Folder to save temporal_average.npy

    Returns
    -------
    ndarray
        Temporal average array

    Notes
    -----
    Mirrors MATLAB logic (align_full_v3_checkpoint.m lines 66-76):
        vid = vid_reader.read_frames(1:vid_reader.frame_count);
        temporal_avg = mean(vid,4);
    """
    from pyflowreg.util.io.factory import get_video_file_reader

    avg_path = output_folder / "temporal_average.npy"

    if avg_path.exists():
        print(f"Loading existing temporal average from {avg_path.name}")
        return np.load(str(avg_path))

    print(f"Computing temporal average from {compensated_path.name}...")
    vid_reader = get_video_file_reader(str(compensated_path))

    # Stream frames to avoid loading entire video into RAM
    frame_count = vid_reader.frame_count

    # Read first frame to get shape
    first_frame = vid_reader.read_frames([0])
    if first_frame.ndim == 4:  # (T, H, W, C)
        first_frame = first_frame[0]  # Remove batch dimension

    # Initialize accumulator
    accumulator = np.zeros_like(first_frame, dtype=np.float64)
    accumulator += first_frame

    # Accumulate remaining frames in batches to avoid RAM spike
    batch_size = 1000  # Process 1000 frames at a time
    for start_idx in range(1, frame_count, batch_size):
        end_idx = min(start_idx + batch_size, frame_count)
        batch_indices = list(range(start_idx, end_idx))
        batch = vid_reader.read_frames(batch_indices)

        # Sum over time axis (axis=0)
        accumulator += np.sum(batch, axis=0, dtype=np.float64)

    # Compute average
    temporal_avg = accumulator / frame_count

    # Save
    np.save(str(avg_path), temporal_avg)
    print(f"Saved temporal average to {avg_path.name}")

    return temporal_avg


def compensate_single_recording(
    input_file: Path,
    config: SessionConfig,
    of_options_override: Optional[Dict] = None,
) -> Path:
    """
    Compensate a single recording with resume support.

    Parameters
    ----------
    input_file : Path
        Path to input recording
    config : SessionConfig
        Session configuration
    of_options_override : dict, optional
        Override specific OFOptions parameters

    Returns
    -------
    Path
        Output folder containing results

    Notes
    -----
    Mirrors MATLAB logic (align_full_v3_checkpoint.m lines 30-79):
    - Skip if compensated.hdf5 exists (line 56-60)
    - Run compensate_recording() with configured parameters (line 58)
    - Compute and cache temporal average (lines 66-76)
    """
    output_root, _ = config.resolve_output_paths()
    output_folder = output_root / input_file.stem

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Check resume
    if config.resume and is_stage1_complete(output_folder):
        print(f"Skipping {input_file.name} - already complete")
        return output_folder

    # Setup OFOptions
    of_params = {
        "input_file": str(input_file),
        "output_path": str(output_folder),
        "output_format": "HDF5",
        "save_valid_idx": True,
        "save_w": False,  # Can be overridden
        "save_meta_info": True,
        "verbose": False,
        "flow_backend": config.flow_backend,
        "backend_params": config.backend_params,
    }

    # Apply overrides if provided
    if of_options_override:
        of_params.update(of_options_override)

    options = OFOptions(**of_params)

    # Run compensation
    print(f"\nStarting motion correction of {input_file.name}...")
    start_time = time()

    # Handle both .hdf5 and .HDF5 (case-insensitive filesystems)
    candidates = [
        output_folder / "compensated.hdf5",
        output_folder / "compensated.HDF5",
    ]
    compensated_h5 = next((p for p in candidates if p.exists()), candidates[0])

    if not compensated_h5.exists():
        compensate_recording(options)
        # Re-check both candidates after compensation
        compensated_h5 = next((p for p in candidates if p.exists()), candidates[0])
    else:
        print(f"Found existing {compensated_h5.name}, skipping compensation")

    # Verify output exists
    if not compensated_h5.exists():
        raise RuntimeError(
            "Compensation failed - missing compensated.hdf5 or compensated.HDF5 after compensate_recording()"
        )

    # Compute temporal average
    compute_and_save_temporal_average(compensated_h5, output_folder)

    # Mark as complete
    status = load_or_create_status(output_folder)
    status["stage1"] = "done"
    save_status(output_folder, status)

    elapsed = time() - start_time
    print(f"Done after {elapsed:.2f} seconds")

    return output_folder


def run_stage1(
    config: SessionConfig,
    of_options_override: Optional[Dict] = None,
    task_index: Optional[int] = None,
) -> List[Path]:
    """
    Run Stage 1 for all or subset of recordings.

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    of_options_override : dict, optional
        Override specific OFOptions parameters for all recordings
    task_index : int, optional
        If provided, process only this recording (for array jobs).
        Uses 0-based indexing.

    Returns
    -------
    list[Path]
        Output folders for processed recordings

    Notes
    -----
    Mirrors MATLAB align_full_v3_checkpoint.m lines 22-80.

    For array jobs, set task_index to select which recording to process.
    The index is 0-based (unlike SLURM/SGE which are 1-based).

    Examples
    --------
    >>> config = SessionConfig.from_toml("session.toml")
    >>> # Process all recordings
    >>> folders = run_stage1(config)
    >>> # Process only recording at index 2 (for array job)
    >>> folder = run_stage1(config, task_index=2)
    """
    complete_script_timer = time()

    # Create output directory
    output_root, final_results = config.resolve_output_paths()
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover inputs
    input_files = discover_input_files(config)
    print(f"Found {len(input_files)} files matching pattern '{config.pattern}'\n")

    # Filter by task index if provided
    if task_index is not None:
        if task_index < 0 or task_index >= len(input_files):
            raise ValueError(
                f"Task index {task_index} out of range [0, {len(input_files)-1}]"
            )
        input_files = [input_files[task_index]]
        print(f"Processing task {task_index}: {input_files[0].name}\n")

    print("Starting Step 1: Motion correction of each sequence...\n")

    # Process each recording
    output_folders = []
    for idx, input_file in enumerate(input_files):
        output_folder = compensate_single_recording(
            input_file, config, of_options_override
        )
        output_folders.append(output_folder)

        # Progress
        remaining = len(input_files) - (idx + 1)
        if remaining > 0:
            print(f"{remaining} file(s) remaining.\n")

    total_time = time() - complete_script_timer
    print(f"\nStage 1 complete. Total time: {total_time:.2f} seconds")

    return output_folders


def run_stage1_array(
    config: SessionConfig, of_options_override: Optional[Dict] = None
) -> Path:
    """
    Run Stage 1 for array job (auto-detect task ID).

    Parameters
    ----------
    config : SessionConfig
        Session configuration
    of_options_override : dict, optional
        Override specific OFOptions parameters

    Returns
    -------
    Path
        Output folder for processed recording

    Raises
    ------
    RuntimeError
        If no array task ID found in environment

    Notes
    -----
    Automatically detects task ID from SLURM_ARRAY_TASK_ID, SGE_TASK_ID,
    or PBS_ARRAY_INDEX environment variables.

    Converts 1-based scheduler indices to 0-based Python indexing.
    """
    task_id = get_array_task_id()

    if task_id is None:
        raise RuntimeError(
            "No array task ID found in environment. "
            "Set SLURM_ARRAY_TASK_ID, SGE_TASK_ID, or PBS_ARRAY_INDEX."
        )

    # Convert to 0-based index (schedulers are typically 1-based)
    task_index = task_id - 1

    print(f"Array task ID: {task_id} (processing index {task_index})")

    folders = run_stage1(config, of_options_override, task_index=task_index)

    return folders[0] if folders else None
