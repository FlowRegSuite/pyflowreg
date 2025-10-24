from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Optional, Tuple, List, Callable, Dict

import numpy as np

from pyflowreg.core.warping import imregister_wrapper
from pyflowreg._runtime import RuntimeContext
from pyflowreg.util.image_processing import normalize, apply_gaussian_filter
from pyflowreg.motion_correction.OF_options import OutputFormat, ChannelNormalization

# Import to trigger executor registration (side effect)
import pyflowreg.motion_correction.parallelization as _parallelization  # noqa: F401


@dataclass
class RegistrationConfig:
    """Simplified configuration."""

    n_jobs: int = -1  # -1 = all cores
    verbose: bool = False
    parallelization: Optional[str] = (
        None  # None = auto-select, or 'sequential', 'threading', 'multiprocessing'
    )


class BatchMotionCorrector:
    """
    Main registration pipeline.
    """

    def __init__(self, options: Any, config: Optional[RegistrationConfig] = None):
        self.options = options
        self.config = config or RegistrationConfig()

        # Statistics
        self.mean_disp: List[float] = []
        self.max_disp: List[float] = []
        self.mean_div: List[float] = []
        self.mean_translation: List[float] = []

        # State
        self.reference_raw: Optional[np.ndarray] = None
        self.reference_proc: Optional[np.ndarray] = None
        self.weight: Optional[np.ndarray] = None
        self.w_init: Optional[np.ndarray] = None

        # I/O
        self.video_reader = None
        self.video_writer = None
        self.w_writer = None
        self.idx_writer = None

        # Progress callbacks
        self.progress_callbacks: List[Callable[[int, int], None]] = []
        # Task-based progress tracking: {task_id: (frames_processed, total_frames)}
        self._progress_trackers: Dict[str, Tuple[int, Optional[int]]] = {}

        # New batch data callbacks
        self.w_callbacks: List[Callable[[np.ndarray, int, int], None]] = []
        self.registered_callbacks: List[Callable[[np.ndarray, int, int], None]] = []

        # Get number of workers
        if self.config.n_jobs == -1:
            import os

            self.n_workers = os.cpu_count() or 4
        else:
            self.n_workers = self.config.n_jobs

        # Initialize executor from RuntimeContext
        self._setup_executor()

        # Resolve displacement function
        self._resolve_displacement_func()

    def _setup_executor(self):
        """Setup the parallelization executor based on configuration."""
        import warnings
        from pyflowreg.core.backend_registry import get_backend_executors

        # Get executor class from RuntimeContext
        executor_name = self.config.parallelization
        backend_name = getattr(self.options, "flow_backend", "flowreg")

        # Get supported executors for this backend
        # Let it raise ValueError if backend is truly unknown
        supported_executors = get_backend_executors(backend_name)

        # Check if requested executor is both supported by backend and available on system
        if executor_name is not None:
            # Get what's actually available on this system
            available = RuntimeContext.get("available_parallelization", set())

            # Check both backend support and system availability
            is_supported = executor_name in supported_executors
            is_available = executor_name in available

            if not is_supported or not is_available:
                # Find intersection of what's supported and what's available
                usable_executors = supported_executors & available

                # Pick the best from what's usable
                fallback_order = ["multiprocessing", "threading", "sequential"]
                fallback = None

                for mode in fallback_order:
                    if mode in usable_executors:
                        fallback = mode
                        break

                if fallback is None:
                    # This should rarely happen - sequential should always be available
                    if "sequential" in supported_executors:
                        fallback = "sequential"
                    else:
                        raise RuntimeError(
                            f"No compatible parallelization executor found. "
                            f"Backend '{backend_name}' supports {sorted(supported_executors)}, "
                            f"but only {sorted(available)} are available on this system."
                        )

                # Determine reason for fallback
                if not is_supported:
                    reason = f"Backend '{backend_name}' does not support '{executor_name}' executor"
                else:
                    reason = (
                        f"Executor '{executor_name}' is not available on this system"
                    )

                warnings.warn(
                    f"{reason}. "
                    f"Supported executors: {sorted(supported_executors)}. "
                    f"Falling back to '{fallback}'."
                )

                if self.options.verbose or self.config.verbose:
                    print(
                        f"Backend '{backend_name}' using '{fallback}' executor "
                        f"(requested '{executor_name}' not usable: {reason})"
                    )

                executor_name = fallback
        else:
            # Auto-select best available that's supported by backend
            available = RuntimeContext.get("available_parallelization", set())

            # Find intersection
            usable_executors = supported_executors & available

            if not usable_executors:
                raise RuntimeError(
                    f"No compatible parallelization executor found. "
                    f"Backend '{backend_name}' supports {sorted(supported_executors)}, "
                    f"but only {sorted(available)} are available on this system."
                )

            # Pick the best from what's usable
            preference_order = ["multiprocessing", "threading", "sequential"]
            for mode in preference_order:
                if mode in usable_executors:
                    executor_name = mode
                    break

        # Get executor class
        executor_class = RuntimeContext.get_parallelization_executor(executor_name)
        if executor_class is None:
            # Fallback to sequential if requested executor not available
            if not self.config.verbose:
                print(
                    f"Warning: {executor_name} executor not available, falling back to sequential"
                )
            executor_class = RuntimeContext.get_parallelization_executor("sequential")

            # If sequential is also not available, import and register it
            if executor_class is None:
                from pyflowreg.motion_correction.parallelization.sequential import (
                    SequentialExecutor,
                )

                SequentialExecutor.register()
                executor_class = RuntimeContext.get_parallelization_executor(
                    "sequential"
                )

                # Final safety check
                if executor_class is None:
                    raise RuntimeError(
                        "Could not load any executor, including sequential fallback"
                    )

        # Create executor instance
        self.executor = executor_class(n_workers=self.n_workers)

        if not self.config.verbose:
            # Use actual executor name and worker count
            actual_workers = self.executor.n_workers
            worker_str = "worker" if actual_workers == 1 else "workers"
            print(
                f"Using {self.executor.name} executor with {actual_workers} {worker_str}"
            )

    def _resolve_displacement_func(self):
        """Resolve the displacement function to use based on options."""
        self._get_disp = self.options.resolve_get_displacement()

    def register_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """
        Register a progress callback function.

        Args:
            callback: Function that receives (current_frame, total_frames) as arguments.
                      For multiprocessing, updates are batch-wise rather than frame-wise.
        """
        if callback not in self.progress_callbacks:
            self.progress_callbacks.append(callback)

    def register_w_callback(
        self, callback: Callable[[np.ndarray, int, int], None]
    ) -> None:
        """
        Register a callback for displacement field batches.

        Args:
            callback: Function receiving (w_batch, batch_start_idx, batch_end_idx)
                      where w_batch has shape (T, H, W, 2) containing [u,v] components.
                      batch_start_idx and batch_end_idx indicate the frame indices in the original video.
        """
        if callback not in self.w_callbacks:
            self.w_callbacks.append(callback)

    def register_registered_callback(
        self, callback: Callable[[np.ndarray, int, int], None]
    ) -> None:
        """
        Register a callback for registered/compensated frame batches.

        Args:
            callback: Function receiving (registered_batch, batch_start_idx, batch_end_idx)
                      where registered_batch has shape (T, H, W, C).
                      batch_start_idx and batch_end_idx indicate the frame indices in the original video.
        """
        if callback not in self.registered_callbacks:
            self.registered_callbacks.append(callback)

    def _notify_progress(self, frames_completed: int, task_id: str = "main") -> None:
        """
        Notify all registered progress callbacks for a specific task.

        Args:
            frames_completed: Number of frames just completed (to add to total)
            task_id: Identifier for the task being tracked (default: "main")
        """
        # Initialize task tracker if needed
        if task_id not in self._progress_trackers:
            # For main task, use the total frames from video reader
            total = self._total_frames if task_id == "main" else None
            self._progress_trackers[task_id] = (0, total)

        # Update progress for this task
        current, total = self._progress_trackers[task_id]
        current += frames_completed
        self._progress_trackers[task_id] = (current, total)

        # Only notify callbacks for the main task
        if task_id == "main" and total and self.progress_callbacks:
            for callback in self.progress_callbacks:
                try:
                    callback(current, total)
                except Exception as e:
                    warnings.warn(f"Progress callback error: {e}")

    def _setup_io(self):
        """Setup I/O handlers."""
        output_path = Path(self.options.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.video_reader = self.options.get_video_reader()
        self.video_writer = self.options.get_video_writer()

        # Create displacement writer if requested
        if getattr(self.options, "save_w", False):
            try:
                from pyflowreg.util.io.factory import get_video_file_writer

                # Use ArrayWriter for displacements when main output is ARRAY
                if self.options.output_format == OutputFormat.ARRAY:
                    self.w_writer = get_video_file_writer(
                        None,  # Path ignored for ARRAY format
                        OutputFormat.ARRAY.value,
                    )
                else:
                    # Use HDF5 for file-based output (preserves double precision)
                    w_path = output_path / "w.h5"
                    self.w_writer = get_video_file_writer(
                        str(w_path), "HDF5", dataset_names=["u", "v"]
                    )
            except Exception as e:
                warnings.warn(
                    f"Failed to create displacement writer: {e}. Displacements will not be saved."
                )
                self.w_writer = None
                self.options.save_w = False  # Disable to avoid trying to write later

        # Create valid mask writer if requested
        if getattr(self.options, "save_valid_idx", False):
            try:
                from pyflowreg.util.io.factory import get_video_file_writer

                # Use HDF5 for idx (MATLAB compatibility: idx.hdf)
                idx_path = output_path / "idx.hdf"
                self.idx_writer = get_video_file_writer(
                    str(idx_path), "HDF5", dataset_name="idx"
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to create valid mask writer: {e}. Valid masks will not be saved."
                )
                self.idx_writer = None
                self.options.save_valid_idx = False

    def _setup_reference(self, reference_frame: Optional[np.ndarray] = None):
        """Setup reference frame and weights."""
        if reference_frame is None:
            self.reference_raw = self.options.get_reference_frame(
                self.video_reader
            ).astype(np.float64)
        else:
            self.reference_raw = reference_frame.astype(np.float64)

        H, W = self.reference_raw.shape[:2]
        n_channels = self.reference_raw.shape[2] if self.reference_raw.ndim == 3 else 1

        # Setup weights
        self.weight = np.ones((H, W, n_channels), dtype=np.float64)
        if hasattr(self.options, "get_weight_at"):
            for c in range(n_channels):
                self.weight[:, :, c] = self.options.get_weight_at(c, n_channels)
        else:
            weight_1d = np.asarray(getattr(self.options, "weight", [1.0] * n_channels))
            weight_sum = weight_1d.sum()
            if weight_sum > 0:
                weight_1d = weight_1d / weight_sum
            for c in range(n_channels):
                self.weight[:, :, c] = (
                    weight_1d[c] if c < len(weight_1d) else 1.0 / n_channels
                )

        # Preprocess reference (MATLAB order: normalize then filter)
        self.reference_proc = self._preprocess_frames(self.reference_raw)

    def _preprocess_frames(
        self, frames: np.ndarray, normalization_ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Preprocess frames: normalize -> filter (MATLAB order).

        Args:
            frames: Frames to preprocess
            normalization_ref: Reference array for normalization. If None, uses frames' own min/max.
                              Should be the raw reference to ensure consistent normalization.
        """
        # First normalize
        # Map enum to expected string: JOINT -> 'together', SEPARATE -> 'separate'
        if hasattr(self.options, "channel_normalization"):
            norm_mode = self.options.channel_normalization
            if norm_mode == ChannelNormalization.JOINT:
                norm_value = "together"
            else:
                norm_value = "separate"
        else:
            norm_value = "together"

        normalized = normalize(
            frames, ref=normalization_ref, channel_normalization=norm_value
        )
        # Then filter
        filtered = apply_gaussian_filter(
            normalized,
            sigma=np.asarray(self.options.sigma),
            mode="reflect",
            truncate=4.0,
        )
        return filtered.astype(np.float64)

    def _compute_flow_single(
        self,
        frame_proc: np.ndarray,
        ref_proc: np.ndarray,
        w_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute flow for a single frame."""
        flow_params = {
            "alpha": self.options.alpha,
            "weight": self.weight,
            "levels": self.options.levels,
            "min_level": getattr(
                self.options,
                "effective_min_level",
                getattr(self.options, "min_level", 0),
            ),
            "eta": self.options.eta,
            "update_lag": self.options.update_lag,
            "iterations": self.options.iterations,
            "a_smooth": self.options.a_smooth,
            "a_data": self.options.a_data,
        }

        if w_init is not None:
            flow_params["uv"] = w_init

        # Note: get_displacement expects (reference, moving)
        return self._get_disp(ref_proc, frame_proc, **flow_params)

    def _process_batch_parallel(
        self,
        batch: np.ndarray,
        batch_proc: np.ndarray,
        w_init: np.ndarray,
        task_id: str = "main",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process batch using the configured executor.

        Args:
            batch: Raw batch of frames
            batch_proc: Preprocessed batch
            w_init: Initial displacement field
            task_id: Task identifier for progress tracking (default: "main")
        """
        # Build flow parameters dictionary
        flow_params = {
            "alpha": self.options.alpha,
            "weight": self.weight,
            "levels": self.options.levels,
            "min_level": getattr(
                self.options,
                "effective_min_level",
                getattr(self.options, "min_level", 0),
            ),
            "eta": self.options.eta,
            "update_lag": self.options.update_lag,
            "iterations": self.options.iterations,
            "a_smooth": self.options.a_smooth,
            "a_data": self.options.a_data,
        }

        # Get interpolation method
        interp_method = getattr(self.options, "interpolation_method", "cubic")

        # Create progress callback wrapper for executor if we have callbacks registered
        # Only track progress for "main" task
        executor_progress_callback = None
        if self.progress_callbacks and task_id == "main":

            def executor_progress_callback(frames_completed: int):
                self._notify_progress(frames_completed, task_id)

        # Use executor to process batch
        return self.executor.process_batch(
            batch=batch,
            batch_proc=batch_proc,
            reference_raw=self.reference_raw,
            reference_proc=self.reference_proc,
            w_init=w_init,
            get_displacement_func=self._get_disp,
            imregister_func=imregister_wrapper,
            interpolation_method=interp_method,
            progress_callback=executor_progress_callback,
            flow_params=flow_params,
        )

    def _compute_initial_w(
        self, first_batch: np.ndarray, first_batch_proc: np.ndarray
    ) -> np.ndarray:
        """Compute initial displacement field from first frames."""
        n_init = min(22, first_batch.shape[0])  # T is first dimension

        if not self.config.verbose:
            print("Computing initial displacement field...")

        # Process first n_init frames - use "initial_w" task to avoid counting toward main progress
        if n_init > 4:  # Use parallel for multiple frames
            _, w = self._process_batch_parallel(
                first_batch[:n_init],
                first_batch_proc[:n_init],
                np.zeros(
                    (self.reference_proc.shape[0], self.reference_proc.shape[1], 2)
                ),
                task_id="initial_w",  # Don't count toward main progress
            )
        else:  # Serial for very few frames
            H, W = self.reference_proc.shape[:2]
            w = np.zeros((n_init, H, W, 2), dtype=np.float32)
            for t in range(n_init):
                w[t] = self._compute_flow_single(
                    first_batch_proc[t], self.reference_proc
                )

        # Average flows
        w_init = np.mean(w, axis=0)

        if not self.config.verbose:
            print("Done pre-registration to get w_init.")

        return w_init

    def _update_reference(self, batch_proc: np.ndarray, w: np.ndarray):
        """Update reference using compensated frames (MATLAB-compatible)."""
        n_ref_frames = min(100, batch_proc.shape[0])  # T is first dimension
        if n_ref_frames < 1:
            return

        # Use last n_ref_frames
        start_idx = batch_proc.shape[0] - n_ref_frames

        # Compensate and average per channel
        H, W, C = self.reference_proc.shape
        new_ref = np.zeros_like(self.reference_proc)

        for c in range(C):
            compensated = np.zeros((n_ref_frames, H, W), dtype=np.float64)
            for t in range(n_ref_frames):
                frame_c = (
                    batch_proc[start_idx + t, :, :, c]
                    if C > 1
                    else batch_proc[start_idx + t, :, :, 0]
                )
                interp_method = getattr(self.options, "interpolation_method", "cubic")
                compensated[t] = imregister_wrapper(
                    frame_c,
                    w[start_idx + t, :, :, 0],
                    w[start_idx + t, :, :, 1],
                    self.reference_proc[:, :, c]
                    if C > 1
                    else self.reference_proc[:, :, 0],
                    interpolation_method=interp_method,
                )
            new_ref[:, :, c] = np.mean(compensated, axis=0)

        self.reference_proc = new_ref

    def run(self, reference_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Run complete registration pipeline."""

        # Setup
        self._setup_io()
        self._setup_reference(reference_frame)

        # Initialize total frames for progress tracking
        self._total_frames = len(self.video_reader) if self.video_reader else None

        if not self.config.verbose:
            quality = getattr(self.options, "quality_setting", "balanced")
            print(f"\nStarting compensation with quality={quality}")
            print(
                f"Buffer size: {self.options.buffer_size}, Workers: {self.executor.n_workers}"
            )

        # Process batches
        batch_idx = 0
        total_frames = 0
        start_time = time()

        try:
            while self.video_reader.has_batch():
                batch_idx += 1
                batch_start = time()

                # Read batch
                batch = self.video_reader.read_batch()  # (T,H,W,C)

                # Preprocess entire batch (normalize -> filter)
                # CRITICAL: Use reference_raw for normalization to ensure consistency!
                batch_proc = self._preprocess_frames(
                    batch, normalization_ref=self.reference_raw
                )

                # First batch: compute w_init
                if batch_idx == 1:
                    self.w_init = self._compute_initial_w(batch, batch_proc)

                # Decide whether to use w_init
                if not getattr(self.options, "update_initialization_w", True):
                    current_w_init = np.zeros_like(self.w_init)
                else:
                    current_w_init = self.w_init

                # Process batch in parallel (progress callbacks handled internally)
                registered, w = self._process_batch_parallel(
                    batch, batch_proc, current_w_init
                )

                # Update w_init for next batch
                if getattr(self.options, "update_initialization_w", True):
                    if w.shape[0] > 20:
                        self.w_init = np.mean(w[-20:], axis=0)
                    else:
                        self.w_init = np.mean(w, axis=0)

                # Compute statistics
                disp_magnitude = np.sqrt(w[:, :, :, 0] ** 2 + w[:, :, :, 1] ** 2)
                self.mean_disp.extend(np.mean(disp_magnitude, axis=(1, 2)).tolist())
                self.max_disp.extend(np.max(disp_magnitude, axis=(1, 2)).tolist())

                # Divergence and translation
                for t in range(w.shape[0]):
                    du_dx = np.gradient(w[t, :, :, 0], axis=1)
                    dv_dy = np.gradient(w[t, :, :, 1], axis=0)
                    self.mean_div.append(float(np.mean(du_dx + dv_dy)))

                    u_mean = float(np.mean(w[t, :, :, 0]))
                    v_mean = float(np.mean(w[t, :, :, 1]))
                    self.mean_translation.append(float(np.sqrt(u_mean**2 + v_mean**2)))

                # Write results
                self.video_writer.write_frames(registered)

                # Notify registered frame callbacks
                batch_start_idx = total_frames
                batch_end_idx = total_frames + registered.shape[0]
                for callback in self.registered_callbacks:
                    try:
                        callback(registered, batch_start_idx, batch_end_idx)
                    except Exception as e:
                        warnings.warn(f"Registered frames callback error: {e}")

                # Save valid masks if requested
                if getattr(self.options, "save_valid_idx", False):
                    if self.idx_writer is not None:
                        from pyflowreg.core.warping import compute_batch_valid_masks

                        valid_batch = compute_batch_valid_masks(w)
                        self.idx_writer.write_frames(valid_batch)
                    else:
                        warnings.warn(
                            "Valid mask saving was requested but writer could not be initialized. Skipping mask save."
                        )

                # Save flows if requested
                if getattr(self.options, "save_w", False):
                    if self.w_writer is not None:
                        # w has shape (T, H, W, 2) where last dimension is [u, v]
                        # Writer with dataset_names=['u', 'v'] will split into separate datasets
                        self.w_writer.write_frames(w)
                    else:
                        warnings.warn(
                            "Displacement saving was requested but writer could not be initialized. Skipping displacement save."
                        )

                # Notify w callbacks (displacement fields)
                for callback in self.w_callbacks:
                    try:
                        callback(w, batch_start_idx, batch_end_idx)
                    except Exception as e:
                        warnings.warn(f"Displacement field callback error: {e}")

                # Update reference if requested
                if getattr(self.options, "update_reference", False):
                    self._update_reference(batch_proc, w)

                # Progress
                total_frames += registered.shape[0]
                batch_time = time() - batch_start

                if not self.config.verbose:
                    fps = registered.shape[0] / batch_time
                    print(
                        f"Batch {batch_idx}: {registered.shape[0]} frames in {batch_time:.2f}s ({fps:.1f} fps)"
                    )

        finally:
            # Cleanup executor
            if self.executor is not None:
                self.executor.cleanup()

        # Final stats
        total_time = time() - start_time
        if not self.config.verbose:
            avg_fps = total_frames / max(1e-6, total_time)
            print(
                f"\nProcessed {total_frames} frames in {total_time:.2f}s (avg {avg_fps:.1f} fps)"
            )

        # Save metadata
        self._save_metadata()

        # Cleanup
        self._cleanup()

        return self.reference_raw

    def _save_metadata(self):
        """Save statistics and reference frame."""
        if not getattr(self.options, "save_meta_info", False):
            return

        output_path = Path(self.options.output_path)

        # Save statistics
        stats_path = output_path / "statistics.npz"
        np.savez(
            str(stats_path),
            mean_disp=np.array(self.mean_disp),
            max_disp=np.array(self.max_disp),
            mean_div=np.array(self.mean_div),
            mean_translation=np.array(self.mean_translation),
        )

        # Save reference
        if self.reference_raw is not None:
            ref_path = output_path / "reference_frame.npy"
            np.save(str(ref_path), self.reference_raw)

        print(f"Saved metadata to {output_path}")

    def _cleanup(self):
        """Close file handlers."""
        if self.video_writer is not None:
            self.video_writer.close()
        if hasattr(self, "w_writer") and self.w_writer is not None:
            self.w_writer.close()
        if hasattr(self, "idx_writer") and self.idx_writer is not None:
            self.idx_writer.close()


def compensate_recording(
    options: Any,
    reference_frame: Optional[np.ndarray] = None,
    config: Optional[RegistrationConfig] = None,
) -> np.ndarray:
    """
    Main entry point matching MATLAB API.

    Args:
        options: OF_options object with parameters
        reference_frame: Optional pre-computed reference
        config: Optional registration configuration

    Returns:
        The reference frame used
    """
    pipeline = BatchMotionCorrector(options, config)
    return pipeline.run(reference_frame)


if __name__ == "__main__":
    try:
        from OF_options import OFOptions

        options = OFOptions(
            input_file="test_video.h5",
            output_path="results",
            quality_setting="balanced",
            save_w=True,
            sigma=[[1.0, 1.0, 0.5], [1.0, 1.0, 0.5]],  # [sx, sy, st] per channel
            update_reference=False,
            update_initialization_w=True,
        )

        config = RegistrationConfig(
            n_jobs=-1,  # Use all cores
        )

        ref = compensate_recording(options, config=config)

    except ImportError:
        print("OF_options not available")
