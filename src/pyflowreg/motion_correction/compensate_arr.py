"""
Array-based motion compensation using the same pipeline as file-based processing.
Provides MATLAB compensate_inplace equivalent functionality.
"""

from typing import Optional, Tuple, Callable, Dict, Any
import numpy as np

from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat
from pyflowreg.motion_correction.compensate_recording import BatchMotionCorrector


def compensate_arr(
    c1: np.ndarray,
    c_ref: np.ndarray,
    options: Optional[OFOptions] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    *,
    flow_backend: Optional[str] = None,
    backend_params: Optional[Dict[str, Any]] = None,
    get_displacement: Optional[Callable] = None,
    get_displacement_factory: Optional[Callable[..., Callable]] = None,
    w_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
    registered_callback: Optional[Callable[[np.ndarray, int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process arrays in memory matching MATLAB compensate_inplace functionality.

    This function provides the same motion compensation as compensate_recording
    but operates on in-memory arrays instead of files. It uses the same batching
    and flow initialization logic to ensure algorithmic consistency.

    Args:
        c1: Input array to register, shape (T,H,W,C), (H,W,C), or (T,H,W)
            For single-channel 3D arrays, assumes (T,H,W) if T > 4, else (H,W,C)
        c_ref: Reference frame, shape (H,W,C) or (H,W)
        options: OF_options configuration. If None, uses defaults.
        progress_callback: Optional callback function that receives (current_frame, total_frames)
            for progress updates. Note: For multiprocessing executor, updates are batch-wise.
        flow_backend: Backend name override (e.g., 'diso', 'flowreg')
        backend_params: Backend-specific parameters override
        get_displacement: Direct displacement callable override
        get_displacement_factory: Factory function override for creating displacement callable
        w_callback: Optional callback for displacement field batches, receives (w_batch, start_idx, end_idx)
        registered_callback: Optional callback for registered frame batches, receives (batch, start_idx, end_idx)

    Returns:
        Tuple of:
            - c_reg: Registered array with same shape as input
            - w: Displacement fields, shape (T,H,W,2) with [u,v] components

    Example:
        >>> import numpy as np
        >>> from pyflowreg.motion_correction import compensate_arr
        >>>
        >>> # Create test data
        >>> video = np.random.rand(100, 256, 256, 2)  # 100 frames, 2 channels
        >>> reference = np.mean(video[:10], axis=0)
        >>>
        >>> # Register with progress callback
        >>> def progress(current, total):
        ...     print(f"Progress: {current}/{total} ({100*current/total:.1f}%)")
        >>> registered, flow = compensate_arr(video, reference, progress_callback=progress)
    """
    # Handle 3D squeeze for single channel (MATLAB compatibility)
    squeezed = False
    original_shape = c1.shape

    # Validate input is not empty
    if c1.size == 0:
        raise ValueError("Input array cannot be empty")

    if c1.ndim == 3 and c_ref.ndim == 2:
        # Input is 3D, reference is 2D - add channel dimension
        c1 = c1[..., np.newaxis]
        c_ref = c_ref[..., np.newaxis]
        squeezed = True
    elif c1.ndim == 2:
        # Single frame, single channel
        c1 = c1[np.newaxis, :, :, np.newaxis]
        if c_ref.ndim == 2:
            c_ref = c_ref[..., np.newaxis]
        squeezed = True

    # Configure options for array processing
    if options is None:
        options = OFOptions()
    else:
        # Make a copy to avoid modifying user's options
        options = options.copy()

    # Apply backend overrides
    if flow_backend is not None:
        options.flow_backend = flow_backend
    if backend_params is not None:
        if options.backend_params:
            options.backend_params.update(backend_params)
        else:
            options.backend_params = backend_params
    if get_displacement is not None:
        options.get_displacement_impl = get_displacement
    if get_displacement_factory is not None:
        options.get_displacement_factory = get_displacement_factory

    # Set up for array I/O
    options.input_file = c1  # Will be wrapped by factory into ArrayReader
    options.reference_frames = c_ref
    options.output_format = (
        OutputFormat.ARRAY
    )  # Triggers ArrayWriter in factory (must be enum value)

    # Enable saving displacement fields to get them back
    options.save_w = True

    # Disable file-based features
    options.save_meta_info = False

    # Run standard pipeline
    compensator = BatchMotionCorrector(options)

    # Register callbacks if provided
    if progress_callback is not None:
        compensator.register_progress_callback(progress_callback)
    if w_callback is not None:
        compensator.register_w_callback(w_callback)
    if registered_callback is not None:
        compensator.register_registered_callback(registered_callback)

    compensator.run()

    # Get results from ArrayWriter
    c_reg = compensator.video_writer.get_array()

    # Get flow fields from the w_writer (which is also an ArrayWriter when output is ARRAY)
    w = None
    if compensator.w_writer is not None:
        w = compensator.w_writer.get_array()

    # TODO: Handle output_typename casting in ArrayWriter instead of here
    # For now, manual casting if specified
    if hasattr(options, "output_typename") and options.output_typename:
        dtype_map = {
            "single": np.float32,
            "double": np.float64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "int16": np.int16,
            "int32": np.int32,
        }
        if options.output_typename in dtype_map:
            c_reg = c_reg.astype(dtype_map[options.output_typename])

    # Squeeze back if needed to match input shape
    if squeezed:
        if len(original_shape) == 2:
            # Was single frame (H,W)
            c_reg = np.squeeze(c_reg)
            if w is not None:
                w = np.squeeze(w, axis=0)  # Remove time dimension
        elif len(original_shape) == 3:
            # Was (T,H,W) or (H,W,C)
            c_reg = np.squeeze(c_reg, axis=-1)  # Remove channel dimension

    # If no flow fields were captured, create empty array
    if w is None:
        if c_reg.ndim >= 3:
            T = c_reg.shape[0] if c_reg.ndim == 4 else 1
            H, W = c_reg.shape[-3:-1] if c_reg.ndim == 4 else c_reg.shape[:2]
        else:
            T, H, W = 1, c_reg.shape[0], c_reg.shape[1]
        w = np.zeros((T, H, W, 2), dtype=np.float32)

    return c_reg, w


def compensate_pair(
    frame1: np.ndarray, frame2: np.ndarray, options: Optional[OFOptions] = None
) -> np.ndarray:
    """
    Compute optical flow between two frames.

    Args:
        frame1: Reference frame, shape (H,W,C) or (H,W)
        frame2: Moving frame to register to frame1, shape (H,W,C) or (H,W)
        options: OF_options configuration. If None, uses defaults.

    Returns:
        w: Displacement field, shape (H,W,2) with [u,v] components
    """
    if frame1.ndim == 2:
        frame1 = frame1[..., np.newaxis]
    if frame2.ndim == 2:
        frame2 = frame2[..., np.newaxis]

    frames = np.stack([frame1, frame2], axis=0)
    _, w = compensate_arr(frames, frame1, options)

    return w[1]
