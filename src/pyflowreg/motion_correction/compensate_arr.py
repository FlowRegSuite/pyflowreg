"""
Array-based motion compensation using the same pipeline as file-based processing.
Provides MATLAB compensate_inplace equivalent functionality.
"""

from typing import Optional, Tuple, Callable, Dict, Any
import numpy as np

from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat
from pyflowreg.motion_correction.compensate_recording import (
    BatchMotionCorrector,
    RegistrationConfig,
)


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
    registration_config: Optional[RegistrationConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register an in-memory array against a reference frame.

    Provides the same motion compensation as ``compensate_recording`` but
    operates on in-memory arrays instead of files (MATLAB
    ``compensate_inplace`` equivalent). The array is routed through the same
    batching and flow-initialization pipeline, with results collected by an
    in-memory writer instead of being written to disk.

    Parameters
    ----------
    c1 : ndarray
        Input array to register, shape (T, H, W, C), (T, H, W), (H, W, C),
        or (H, W). A 3D input is treated as (T, H, W) when ``c_ref`` is 2D;
        otherwise it is treated as a single (H, W, C) frame when the last
        dimension is at most 4 (the ArrayReader convention), and as
        (T, H, W) when it is larger.
    c_ref : ndarray
        Reference frame, shape (H, W, C) or (H, W).
    options : OFOptions, optional
        Optical flow configuration. If None, default ``OFOptions`` are used.
        The function works on a copy and forces ``output_format`` to ARRAY
        (any user-set ``output_format`` is ignored), ``save_w=True``, and
        ``save_meta_info=False``.
    progress_callback : callable, optional
        Called as ``progress_callback(current_frame, total_frames)`` for
        progress updates. For the multiprocessing executor, updates are
        batch-wise rather than frame-wise.
    flow_backend : str, optional
        Flow backend name override (e.g., 'diso', 'flowreg'); sets
        ``options.flow_backend`` on the internal copy.
    backend_params : dict, optional
        Backend-specific parameter overrides, merged into
        ``options.backend_params``.
    get_displacement : callable, optional
        Direct displacement callable override; sets
        ``options.get_displacement_impl``.
    get_displacement_factory : callable, optional
        Factory override for creating the displacement callable; sets
        ``options.get_displacement_factory``.
    w_callback : callable, optional
        Called as ``w_callback(w_batch, start_idx, end_idx)`` for each
        batch of displacement fields, where ``w_batch`` has shape
        (T, H, W, 2).
    registered_callback : callable, optional
        Called as ``registered_callback(batch, start_idx, end_idx)`` for
        each batch of registered frames, where ``batch`` has shape
        (T, H, W, C).
    registration_config : RegistrationConfig, optional
        Execution configuration (parallelization mode, worker count,
        verbosity).

    Returns
    -------
    c_reg : ndarray
        Registered frames, with the same shape (rank) as the input:
        (T, H, W, C), (T, H, W), (H, W, C), or (H, W). Cast according to
        ``options.output_typename`` (default "double", i.e. float64) when
        it is one of "single", "double", "uint8", "uint16", "int16",
        "int32"; other values leave the pipeline output dtype unchanged.
    w : ndarray
        Displacement fields, shape (T, H, W, 2) with components (u, v),
        where ``w[..., 0]`` is the horizontal (x) and ``w[..., 1]`` the
        vertical (y) displacement. For single-frame inputs ((H, W) or
        (H, W, C)) the leading time axis is removed, giving (H, W, 2). A
        zero-filled array is returned if no displacement fields were
        captured.

    Raises
    ------
    ValueError
        If ``c1`` is empty.

    See Also
    --------
    compensate_pair : Displacement field between two single frames.
    pyflowreg.motion_correction.compensate_recording.compensate_recording :
        File-based motion correction pipeline.

    Examples
    --------
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
    >>> registered, flow = compensate_arr(video, reference, progress_callback=progress)  # doctest: +SKIP
    """
    # Handle 3D squeeze for single channel (MATLAB compatibility)
    squeezed = False  # channel axis was added
    single_frame = False  # time axis was added (input had no T axis)
    original_shape = c1.shape

    # Validate input is not empty
    if c1.size == 0:
        raise ValueError("Input array cannot be empty")

    if c1.ndim == 3 and c_ref.ndim == 2:
        # Input is 3D, reference is 2D - add channel dimension
        c1 = c1[..., np.newaxis]
        c_ref = c_ref[..., np.newaxis]
        squeezed = True
    elif c1.ndim == 3 and c1.shape[-1] <= 4:
        # Single (H, W, C) frame with a multi-channel reference - add the
        # time axis here (mirrors the ArrayReader heuristic: a 3D array
        # with last dimension <= 4 is one multi-channel frame) so the
        # output rank can mirror the input rank.
        c1 = c1[np.newaxis, ...]
        single_frame = True
    elif c1.ndim == 2:
        # Single frame, single channel
        c1 = c1[np.newaxis, :, :, np.newaxis]
        if c_ref.ndim == 2:
            c_ref = c_ref[..., np.newaxis]
        squeezed = True
        single_frame = True

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
            # Defensive copy so the internal options never alias the
            # caller's kwarg dict.
            options.backend_params = dict(backend_params)
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
    compensator = BatchMotionCorrector(options, config=registration_config)

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

    # Squeeze back so the output rank mirrors the input rank
    if squeezed:
        if len(original_shape) == 2:
            # Was single frame (H,W): drop time and channel axes
            c_reg = np.squeeze(c_reg, axis=(0, -1))
        elif len(original_shape) == 3:
            # Was (T,H,W) with a 2D reference: drop the channel axis
            c_reg = np.squeeze(c_reg, axis=-1)
    elif single_frame:
        # Was a single (H,W,C) frame: drop the time axis
        c_reg = np.squeeze(c_reg, axis=0)

    if w is not None and single_frame:
        w = np.squeeze(w, axis=0)  # Remove time dimension

    # If no flow fields were captured, create a zero-filled array matching
    # the documented shape: (H, W, 2) for single-frame inputs, (T, H, W, 2)
    # otherwise
    if w is None:
        if single_frame:
            H, W = original_shape[:2]
            w = np.zeros((H, W, 2), dtype=np.float32)
        else:
            # (T, H, W, C) or (T, H, W): spatial size at axes 1-2
            H, W = original_shape[1:3]
            w = np.zeros((original_shape[0], H, W, 2), dtype=np.float32)

    return c_reg, w


def compensate_pair(
    frame1: np.ndarray, frame2: np.ndarray, options: Optional[OFOptions] = None
) -> np.ndarray:
    """
    Compute the displacement field between two frames.

    Stacks ``frame1`` and ``frame2`` into a two-frame sequence, runs
    ``compensate_arr`` with ``frame1`` as the reference, and returns only the
    displacement field estimated for ``frame2``. Backward-warping ``frame2``
    with the returned field aligns it to ``frame1``.

    Parameters
    ----------
    frame1 : ndarray
        Reference (fixed) frame, shape (H, W, C) or (H, W).
    frame2 : ndarray
        Moving frame to register to ``frame1``, shape (H, W, C) or (H, W).
    options : OFOptions, optional
        Optical flow configuration forwarded to ``compensate_arr``. If None,
        defaults are used.

    Returns
    -------
    w : ndarray
        Displacement field, shape (H, W, 2) with components (u, v), where
        ``w[..., 0]`` is the horizontal (x) and ``w[..., 1]`` the vertical
        (y) displacement.

    See Also
    --------
    compensate_arr : Register a full array against a reference frame.

    Notes
    -----
    Only the displacement field is returned, not the registered frame. The
    field for ``frame1`` against itself is computed as part of the two-frame
    sequence but discarded.
    """
    if frame1.ndim == 2:
        frame1 = frame1[..., np.newaxis]
    if frame2.ndim == 2:
        frame2 = frame2[..., np.newaxis]

    frames = np.stack([frame1, frame2], axis=0)
    _, w = compensate_arr(frames, frame1, options)

    return w[1]
