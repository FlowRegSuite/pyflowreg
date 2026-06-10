"""
Stage-based z-alignment pipeline.

This module ports the MATLAB patch-based z-shift workflow into the existing
PyFlowReg architecture:

1) Build/load a compensated reference volume.
2) Estimate per-frame/per-pixel z-shifts and optionally write z-corrected data.
3) Optionally simulate a z-shift-only recording from the estimated z-shifts.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.motion_correction.compensate_recording import compensate_recording
from pyflowreg.util.io.factory import get_video_file_reader, get_video_file_writer
from pyflowreg.z_align.config import ZAlignConfig


_STAGE1_PROTECTED_OF_FIELDS = {
    "input_file",
    "output_path",
    "output_format",
    "output_file_name",
    "naming_convention",
    "reference_frames",
}

_RECORDING_PREALIGN_PROTECTED_OF_FIELDS = {
    "input_file",
    "output_path",
    "output_format",
    "output_file_name",
    "naming_convention",
}


def load_or_create_status(output_root: Path) -> Dict[str, Any]:
    """Load ``status.json`` from ``output_root`` or return an empty dict."""
    status_path = output_root / "status.json"
    if status_path.exists():
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_status(output_root: Path, status: Dict[str, Any]) -> None:
    """Atomically persist ``status.json``."""
    status_path = output_root / "status.json"
    tmp_path = status_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    tmp_path.replace(status_path)


def _ensure_thwc(arr: np.ndarray) -> np.ndarray:
    """Normalize frame arrays to THWC layout."""
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        # Either (T,H,W) or (H,W,C) single frame. Treat as (T,H,W) here.
        return arr[:, :, :, np.newaxis]
    if arr.ndim == 2:
        return arr[np.newaxis, :, :, np.newaxis]
    raise ValueError(f"Expected 2D/3D/4D frame array, got {arr.ndim}D")


def _to_hwcz(volume_thwc: np.ndarray) -> np.ndarray:
    """Convert THWC -> HWCZ."""
    return np.transpose(volume_thwc, (1, 2, 3, 0))


def _to_hwct(batch_thwc: np.ndarray) -> np.ndarray:
    """Convert THWC -> HWCT."""
    return np.transpose(batch_thwc, (1, 2, 3, 0))


def _from_hwct(batch_hwct: np.ndarray) -> np.ndarray:
    """Convert HWCT -> THWC."""
    return np.transpose(batch_hwct, (3, 0, 1, 2))


def _iter_batches_with_temporal_halo(reader, halo: int):
    """
    Yield reader batches extended with temporal context from adjacent batches.

    Yields tuples ``(batch_thwc, n_left, n_core)`` where
    ``batch_thwc[n_left:n_left + n_core]`` are the frames of the original
    reader batch and the surrounding frames are up to ``halo`` true context
    frames copied from the neighboring batches. The context lets temporal
    filters see real neighbor frames at batch boundaries instead of their
    reflected boundary handling, so filtered core frames match
    whole-recording processing. At the start and end of the recording no
    context exists, matching the boundary handling a whole-recording filter
    would apply there. Context is limited to the immediately adjacent batch,
    so ``halo`` should not exceed the reader buffer size.
    """
    if halo < 1:
        while reader.has_batch():
            batch = _ensure_thwc(reader.read_batch())
            yield batch, 0, batch.shape[0]
        return

    pending: Optional[np.ndarray] = None
    left_context: Optional[np.ndarray] = None

    while reader.has_batch():
        batch = _ensure_thwc(reader.read_batch())
        if pending is not None:
            parts = [left_context, pending, batch[:halo]]
            extended = np.concatenate([p for p in parts if p is not None], axis=0)
            n_left = 0 if left_context is None else left_context.shape[0]
            yield extended, n_left, pending.shape[0]
            left_context = pending[-halo:]
        pending = batch

    if pending is not None:
        parts = [left_context, pending]
        extended = np.concatenate([p for p in parts if p is not None], axis=0)
        n_left = 0 if left_context is None else left_context.shape[0]
        yield extended, n_left, pending.shape[0]


def _parse_output_format(path: Path, fallback: str = "TIFF") -> str:
    """Infer writer format from file extension."""
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        return "TIFF"
    if ext in {".h5", ".hdf5", ".hdf"}:
        return "HDF5"
    if ext == ".mat":
        return "MAT"
    return fallback


def _resolve_n_workers(n_jobs: int) -> int:
    """Resolve worker count from config style semantics."""
    if n_jobs == -1:
        return os.cpu_count() or 4
    return max(1, int(n_jobs))


def _clip_and_cast(frames: np.ndarray, dtype_name: str) -> np.ndarray:
    """Clip to dtype range and cast. Floating-point outputs are not clipped."""
    dtype = np.dtype(dtype_name)
    arr = frames
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = np.clip(arr, info.min, info.max)
        arr = np.rint(arr)
    return arr.astype(dtype, copy=False)


def _compute_reference_from_source(
    config: ZAlignConfig, source_path: Optional[Path] = None
) -> Optional[np.ndarray]:
    """
    Build a reference image from the configured or provided reference source.

    Mirrors the MATLAB step:
    ``reference = mean(reader.read_frames(1:N), 4)``.
    """
    ref_source = (
        source_path
        if source_path is not None
        else config.resolve_reference_source_file()
    )
    if ref_source is None:
        return None
    if not ref_source.exists():
        raise FileNotFoundError(f"Reference source file not found: {ref_source}")

    reader = get_video_file_reader(
        str(ref_source),
        buffer_size=config.reference_source_buffer_size,
        bin_size=config.reference_source_bin_size,
    )
    try:
        n_frames = min(config.reference_source_frames, len(reader))
        if n_frames < 1:
            raise ValueError("reference_source_file has no frames")
        frames = reader[slice(0, n_frames)]
        frames = _ensure_thwc(frames).astype(np.float32, copy=False)
        reference = np.mean(frames, axis=0)
        return reference
    finally:
        reader.close()


def _build_stage1_overrides(
    config: ZAlignConfig, runtime_override: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge config-level and runtime OFOptions overrides."""
    overrides: Dict[str, Any] = {}
    config_override = config.get_stage1_overrides()
    if config_override:
        overrides.update(config_override)
    if runtime_override:
        overrides.update(runtime_override)
    for field in _STAGE1_PROTECTED_OF_FIELDS:
        overrides.pop(field, None)
    return overrides


def _build_recording_prealign_overrides(config: ZAlignConfig) -> Dict[str, Any]:
    """Return workflow-safe OFOptions overrides for recording prealignment."""
    overrides = config.get_recording_prealign_overrides()
    for field in _RECORDING_PREALIGN_PROTECTED_OF_FIELDS:
        overrides.pop(field, None)
    return overrides


def _compute_xy_gradient(img_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference style 2D gradients (gx, gy)."""
    gy, gx = np.gradient(img_2d.astype(np.float32), edge_order=1)
    return gx.astype(np.float32, copy=False), gy.astype(np.float32, copy=False)


def _compute_volume_gradients(
    volume_hwcz: np.ndarray, spatial_sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-slice spatial gradients for the reference volume."""
    H, W, C, Z = volume_hwcz.shape
    gx_vol = np.zeros((H, W, C, Z), dtype=np.float32)
    gy_vol = np.zeros((H, W, C, Z), dtype=np.float32)

    for c in range(C):
        for z in range(Z):
            smooth = gaussian_filter(volume_hwcz[:, :, c, z], sigma=spatial_sigma)
            gx, gy = _compute_xy_gradient(smooth)
            gx_vol[:, :, c, z] = gx
            gy_vol[:, :, c, z] = gy

    return gx_vol, gy_vol


def _compute_batch_gradients(
    batch_hwct: np.ndarray,
    spatial_sigma: float,
    temporal_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatiotemporal-smoothed gradients for an input batch."""
    H, W, C, T = batch_hwct.shape
    gx_f = np.zeros((H, W, C, T), dtype=np.float32)
    gy_f = np.zeros((H, W, C, T), dtype=np.float32)

    for c in range(C):
        fc = batch_hwct[:, :, c, :]
        fs3 = gaussian_filter(fc, sigma=(spatial_sigma, spatial_sigma, temporal_sigma))
        gy, gx = np.gradient(
            fs3.astype(np.float32, copy=False), axis=(0, 1), edge_order=1
        )
        gx_f[:, :, c, :] = gx.astype(np.float32, copy=False)
        gy_f[:, :, c, :] = gy.astype(np.float32, copy=False)

    return gx_f, gy_f


def _estimate_anchor_z(
    gx_vol: np.ndarray,
    gy_vol: np.ndarray,
    gx_f: np.ndarray,
    gy_f: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Estimate anchor z-index (0-based) from the first batch."""
    Z = gx_vol.shape[3]
    e_sum = np.zeros((Z,), dtype=np.float64)
    for z in range(Z):
        ex = np.abs(gx_vol[:, :, :, z][:, :, :, None] - gx_f)
        ey = np.abs(gy_vol[:, :, :, z][:, :, :, None] - gy_f)
        e_sum[z] = np.sum(ex + ey, dtype=np.float64)
    anchor_z = int(np.argmin(e_sum))
    return anchor_z, e_sum


def _generate_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    """Generate patch starts with guaranteed end coverage."""
    last = max(length - patch_size, 0)
    starts = list(range(0, last + 1, stride))
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def _score_patch(
    patch_bounds: tuple[int, int, int, int],
    gx_vol: np.ndarray,
    gy_vol: np.ndarray,
    gx_f: np.ndarray,
    gy_f: np.ndarray,
    z_candidates: np.ndarray,
    z_indices: np.ndarray,
    z_min: int,
    z_max: int,
    tau_scale: float,
) -> np.ndarray:
    """
    Score one spatial patch against z candidates and return z_hat for all frames.

    Returns
    -------
    ndarray, shape (T,)
        Sub-voxel z estimates for this patch and each time sample.
    """
    r1, r2, c1, c2 = patch_bounds

    gx_patch = gx_f[r1:r2, c1:c2, :, :]
    gy_patch = gy_f[r1:r2, c1:c2, :, :]
    gx_vol_patch = gx_vol[r1:r2, c1:c2, :, :]
    gy_vol_patch = gy_vol[r1:r2, c1:c2, :, :]
    T = gx_patch.shape[3]

    e_patch = np.zeros((T, len(z_candidates)), dtype=np.float64)
    for ii, z in enumerate(z_indices):
        ex = np.abs(gx_vol_patch[:, :, :, z][:, :, :, None] - gx_patch)
        ey = np.abs(gy_vol_patch[:, :, :, z][:, :, :, None] - gy_patch)
        e_patch[:, ii] = np.sum(ex + ey, axis=(0, 1, 2), dtype=np.float64)

    s_patch = -e_patch
    k_rel = np.argmax(s_patch, axis=1)
    km1 = np.maximum(k_rel - 1, 0)
    kp1 = np.minimum(k_rel + 1, len(z_candidates) - 1)
    t_idx = np.arange(T)

    s0 = s_patch[t_idx, k_rel]
    sm = s_patch[t_idx, km1]
    sp = s_patch[t_idx, kp1]
    den = sm - (2.0 * s0) + sp

    tau = tau_scale * np.maximum(np.abs(s0), 1.0)
    den[np.abs(den) < tau] = np.nan

    delta = 0.5 * (sm - sp) / den
    delta[~np.isfinite(delta)] = 0.0
    delta = np.clip(delta, -0.5, 0.5)

    z_hat_patch = np.clip(z_candidates[k_rel] + delta, z_min, z_max)
    return z_hat_patch


def _estimate_z_patchwise(
    gx_vol: np.ndarray,
    gy_vol: np.ndarray,
    gx_f: np.ndarray,
    gy_f: np.ndarray,
    *,
    anchor_z: int,
    win_half: int,
    patch_size: int,
    overlap: float,
    tau_scale: float,
    z_smooth_sigma_spatial: float,
    z_smooth_sigma_temporal: float,
    parallelization: str = "sequential",
    n_jobs: int = -1,
) -> np.ndarray:
    """Patch-based z estimation with sub-voxel quadratic refinement."""
    H, W, _, T = gx_f.shape
    Z = gx_vol.shape[3]

    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    z_min = max(0, anchor_z - win_half)
    z_max = min(Z - 1, anchor_z + win_half)
    z_candidates = np.arange(z_min, z_max + 1, dtype=np.float64)
    z_indices = z_candidates.astype(np.int32, copy=False)

    row_starts = _generate_patch_starts(H, patch_size, stride)
    col_starts = _generate_patch_starts(W, patch_size, stride)
    patches: list[tuple[int, int, int, int]] = []
    for r1 in row_starts:
        r2 = min(H, r1 + patch_size)
        for c1 in col_starts:
            c2 = min(W, c1 + patch_size)
            patches.append((r1, r2, c1, c2))

    z_accum = np.zeros((H, W, T), dtype=np.float64)
    w_accum = np.zeros((H, W, T), dtype=np.float64)

    n_patches = len(patches)
    patch_scores: list[Optional[np.ndarray]] = [None] * n_patches
    n_workers = min(_resolve_n_workers(n_jobs), n_patches) if n_patches else 1
    use_threading = parallelization == "threading" and n_workers > 1

    if use_threading:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _score_patch,
                    patch,
                    gx_vol,
                    gy_vol,
                    gx_f,
                    gy_f,
                    z_candidates,
                    z_indices,
                    z_min,
                    z_max,
                    tau_scale,
                ): patch_idx
                for patch_idx, patch in enumerate(patches)
            }
            for future in as_completed(future_to_idx):
                patch_idx = future_to_idx[future]
                patch_scores[patch_idx] = future.result()
    else:
        for patch_idx, patch in enumerate(patches):
            patch_scores[patch_idx] = _score_patch(
                patch,
                gx_vol,
                gy_vol,
                gx_f,
                gy_f,
                z_candidates,
                z_indices,
                z_min,
                z_max,
                tau_scale,
            )

    # Deterministic accumulation: always add patch contributions in row-major order.
    for patch_idx, (r1, r2, c1, c2) in enumerate(patches):
        z_hat_patch = patch_scores[patch_idx]
        if z_hat_patch is None:
            raise RuntimeError("Missing patch score during z estimation")
        z_accum[r1:r2, c1:c2, :] += z_hat_patch[np.newaxis, np.newaxis, :]
        w_accum[r1:r2, c1:c2, :] += 1.0

    z_hat = z_accum / np.maximum(w_accum, np.finfo(np.float64).eps)
    z_hat = gaussian_filter(
        z_hat,
        sigma=(
            z_smooth_sigma_spatial,
            z_smooth_sigma_spatial,
            z_smooth_sigma_temporal,
        ),
    )
    return np.clip(z_hat, z_min, z_max)


def _apply_z_correction(
    batch_hwct: np.ndarray,
    z_hat_hwt: np.ndarray,
    diff_hwcz: np.ndarray,
) -> np.ndarray:
    """Apply direct z-correction via interpolated ``Diff(anchor)-Diff(z)``."""
    H, W, C, T = batch_hwct.shape
    Z = diff_hwcz.shape[3]
    corrected = np.zeros_like(batch_hwct, dtype=np.float32)
    one = np.float32(1.0)

    for t in range(T):
        zh = np.clip(z_hat_hwt[:, :, t], 0.0, float(Z - 1))
        z0 = np.floor(zh).astype(np.int32)
        z1 = np.minimum(z0 + 1, Z - 1)
        alpha = (zh - z0).astype(np.float32)

        for c in range(C):
            diff_c = diff_hwcz[:, :, c, :]
            d0 = np.take_along_axis(diff_c, z0[:, :, None], axis=2)[:, :, 0]
            d1 = np.take_along_axis(diff_c, z1[:, :, None], axis=2)[:, :, 0]
            corr = (one - alpha) * d0 + alpha * d1
            corrected[:, :, c, t] = batch_hwct[:, :, c, t] + corr

    return corrected


def _simulate_from_z(volume_hwcz: np.ndarray, z_hat_hwt: np.ndarray) -> np.ndarray:
    """Simulate recording frames by interpolating along z in the reference volume."""
    H, W, C, Z = volume_hwcz.shape
    T = z_hat_hwt.shape[2]
    simulated = np.zeros((H, W, C, T), dtype=np.float32)
    one = np.float32(1.0)

    for t in range(T):
        zh = np.clip(z_hat_hwt[:, :, t], 0.0, float(Z - 1))
        z0 = np.floor(zh).astype(np.int32)
        z1 = np.minimum(z0 + 1, Z - 1)
        alpha = (zh - z0).astype(np.float32)
        alpha[z0 == (Z - 1)] = 0.0

        for c in range(C):
            vol_c = volume_hwcz[:, :, c, :]
            v0 = np.take_along_axis(vol_c, z0[:, :, None], axis=2)[:, :, 0]
            v1 = np.take_along_axis(vol_c, z1[:, :, None], axis=2)[:, :, 0]
            simulated[:, :, c, t] = (one - alpha) * v0 + alpha * v1

    return simulated


def _find_compensated_volume(volume_dir: Path) -> Path:
    """Return existing compensated volume path or default target path."""
    candidates = [
        volume_dir / "compensated.HDF5",
        volume_dir / "compensated.hdf5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_volume(config: ZAlignConfig, volume_path: Path) -> np.ndarray:
    """Load reference volume from file and convert to HWCZ float32."""
    reader = get_video_file_reader(
        str(volume_path),
        buffer_size=config.volume_buffer_size,
        bin_size=config.effective_volume_bin_size(),
    )
    try:
        volume_thwc = _ensure_thwc(reader[:]).astype(np.float32, copy=False)
    finally:
        reader.close()

    if volume_thwc.shape[0] < 2:
        raise ValueError("Reference volume must contain at least 2 z slices")
    return _to_hwcz(volume_thwc)


def run_stage1(
    config: ZAlignConfig,
    of_options_override: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Stage 1: build or load the compensated reference volume.

    Returns
    -------
    Path
        Path to reference volume file.
    """
    start_time = time()
    output_root = config.resolve_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    status = load_or_create_status(output_root)
    volume_output_dir = config.resolve_volume_output_dir()
    volume_output_dir.mkdir(parents=True, exist_ok=True)

    if config.reference_volume is not None:
        volume_path = config.resolve_reference_volume_path()
        if not volume_path.exists():
            raise FileNotFoundError(
                f"Configured reference_volume not found: {volume_path}"
            )
        status["stage1"] = "done"
        status["volume_path"] = str(volume_path)
        save_status(output_root, status)
        print(f"Stage 1: using existing reference volume {volume_path}")
        return volume_path

    volume_input_file = config.resolve_volume_input_file()
    if volume_input_file is None:
        raise ValueError(
            "volume_input_file is required when reference_volume is not provided"
        )
    if not volume_input_file.exists():
        raise FileNotFoundError(f"volume_input_file not found: {volume_input_file}")

    if not config.prealign_stack:
        status["stage1"] = "done"
        status["volume_path"] = str(volume_input_file)
        status["prealign_stack"] = False
        save_status(output_root, status)
        print(f"Stage 1: using raw reference stack {volume_input_file}")
        return volume_input_file

    expected_volume = _find_compensated_volume(volume_output_dir)
    if config.resume and status.get("stage1") == "done" and expected_volume.exists():
        print(f"Stage 1: reusing existing volume {expected_volume}")
        return expected_volume

    reference = _compute_reference_from_source(config)
    stage1_buffer_size = config.stack_scans_per_slice or config.stage1_buffer_size
    stage1_update_reference = (
        True
        if config.stack_scans_per_slice is not None
        else config.stage1_update_reference
    )

    of_params: Dict[str, Any] = {
        "input_file": str(volume_input_file),
        "output_path": str(volume_output_dir),
        "output_format": "HDF5",
        "alpha": config.stage1_alpha,
        "quality_setting": config.stage1_quality_setting,
        "buffer_size": stage1_buffer_size,
        "bin_size": config.stage1_bin_size,
        "update_reference": stage1_update_reference,
        "flow_backend": config.flow_backend,
        "backend_params": config.backend_params,
    }
    if reference is not None:
        of_params["reference_frames"] = reference

    overrides = _build_stage1_overrides(config, of_options_override)
    of_params.update(overrides)
    if config.stack_scans_per_slice is not None:
        of_params["buffer_size"] = config.stack_scans_per_slice
        of_params["update_reference"] = True

    options = OFOptions(**of_params)
    print("Stage 1: running compensate_recording to build reference volume...")
    compensate_recording(options)

    volume_path = _find_compensated_volume(volume_output_dir)
    if not volume_path.exists():
        raise RuntimeError(
            "Stage 1 did not produce compensated volume. Expected "
            f"{volume_output_dir / 'compensated.HDF5'}"
        )

    status["stage1"] = "done"
    status["volume_path"] = str(volume_path)
    save_status(output_root, status)

    elapsed = time() - start_time
    print(f"Stage 1 complete in {elapsed:.2f}s")
    return volume_path


def run_recording_prealignment(config: ZAlignConfig) -> Optional[Path]:
    """
    Optionally prealign the Stage-2 recording before z-shift estimation.

    Returns
    -------
    Path or None
        Path to the prealigned recording, or None when disabled.
    """
    if not config.prealign_recording:
        return None

    start_time = time()
    output_root = config.resolve_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    status = load_or_create_status(output_root)

    output_dir = config.resolve_recording_prealigned_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    prealigned_path = config.resolve_recording_prealigned_file()

    if (
        config.resume
        and status.get("recording_prealign") == "done"
        and prealigned_path.exists()
    ):
        print(f"Recording prealignment: reusing {prealigned_path}")
        return prealigned_path

    input_path = config.resolve_input_file()
    if not input_path.exists():
        raise FileNotFoundError(f"input_file not found: {input_path}")

    reference_source = config.resolve_reference_source_file() or input_path
    reference = _compute_reference_from_source(config, reference_source)

    of_params: Dict[str, Any] = {
        "input_file": str(input_path),
        "output_path": str(output_dir),
        "output_format": "HDF5",
        "alpha": config.stage1_alpha,
        "quality_setting": config.stage1_quality_setting,
        "buffer_size": config.input_buffer_size,
        "bin_size": config.input_bin_size,
        "update_reference": False,
        "flow_backend": config.flow_backend,
        "backend_params": config.backend_params,
    }
    if reference is not None:
        of_params["reference_frames"] = reference

    of_params.update(_build_recording_prealign_overrides(config))

    options = OFOptions(**of_params)
    print("Recording prealignment: running compensate_recording...")
    compensate_recording(options)

    if not prealigned_path.exists():
        raise RuntimeError(
            "Recording prealignment did not produce output. Expected "
            f"{prealigned_path}"
        )

    status["recording_prealign"] = "done"
    status["prealigned_recording_path"] = str(prealigned_path)
    save_status(output_root, status)

    elapsed = time() - start_time
    print(f"Recording prealignment complete in {elapsed:.2f}s")
    return prealigned_path


def run_stage2(
    config: ZAlignConfig,
    volume_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Stage 2: estimate z-shifts and optionally write z-corrected output.

    Returns
    -------
    dict
        Keys: ``z_shift_path``, ``corrected_path``, ``anchor_z``, and
        ``prealigned_recording_path``.
    """
    start_time = time()
    output_root = config.resolve_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    status = load_or_create_status(output_root)

    if volume_path is None:
        volume_path = config.resolve_reference_volume_path()
    if not volume_path.exists():
        raise FileNotFoundError(f"Reference volume not found: {volume_path}")

    z_shift_path = config.resolve_z_shift_file()
    corrected_path = config.resolve_corrected_output_file()
    z_shift_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_path.parent.mkdir(parents=True, exist_ok=True)

    stage2_outputs_ready = z_shift_path.exists() and (
        (not config.write_corrected) or corrected_path.exists()
    )
    if config.resume and status.get("stage2") == "done" and stage2_outputs_ready:
        prealigned_recording_path = (
            run_recording_prealignment(config) if config.prealign_recording else None
        )
        anchor_z = status.get("anchor_z", None)
        print("Stage 2: existing outputs found, skipping")
        return {
            "z_shift_path": z_shift_path,
            "corrected_path": corrected_path if config.write_corrected else None,
            "anchor_z": anchor_z,
            "prealigned_recording_path": prealigned_recording_path,
        }

    prealigned_recording_path = run_recording_prealignment(config)
    input_path = prealigned_recording_path or config.resolve_input_file()
    if not input_path.exists():
        raise FileNotFoundError(f"input_file not found: {input_path}")

    volume_hwcz = _load_volume(config, volume_path)
    H, W, C, Z = volume_hwcz.shape
    gx_vol, gy_vol = _compute_volume_gradients(volume_hwcz, config.spatial_sigma)

    # The prealigned recording was already temporally binned by
    # compensate_recording, so it must not be binned a second time here.
    input_bin_size = (
        1 if prealigned_recording_path is not None else config.input_bin_size
    )
    input_reader = get_video_file_reader(
        str(input_path),
        buffer_size=config.input_buffer_size,
        bin_size=input_bin_size,
    )

    z_writer = get_video_file_writer(str(z_shift_path), "HDF5")
    corrected_writer = None
    if config.write_corrected:
        corrected_fmt = _parse_output_format(corrected_path, fallback="TIFF")
        corrected_writer = get_video_file_writer(str(corrected_path), corrected_fmt)

    anchor_z: Optional[int] = None
    diff_hwcz: Optional[np.ndarray] = None

    # Temporal context so the two temporal Gaussian filters (gradient
    # pre-smoothing and z_hat regularization) see true neighbor frames at
    # batch boundaries. The sum of both kernel radii (scipy truncates at
    # 4*sigma) makes the cascaded filtering of core frames equivalent to
    # whole-recording processing, so z estimates do not depend on
    # input_buffer_size.
    temporal_halo = int(np.ceil(4.0 * config.temporal_sigma)) + int(
        np.ceil(4.0 * config.z_smooth_sigma_temporal)
    )

    try:
        n_batches = 0
        for batch_thwc, n_left, n_core in _iter_batches_with_temporal_halo(
            input_reader, temporal_halo
        ):
            batch_hwct = _to_hwct(batch_thwc).astype(np.float32, copy=False)
            if batch_hwct.shape[:3] != (H, W, C):
                input_shape = (
                    batch_hwct.shape[0],
                    batch_hwct.shape[1],
                    batch_hwct.shape[2],
                )
                raise ValueError(
                    "Input recording dimensions do not match reference volume: "
                    f"input {input_shape} "
                    f"vs volume {(H, W, C)}"
                )

            gx_f, gy_f = _compute_batch_gradients(
                batch_hwct,
                spatial_sigma=config.spatial_sigma,
                temporal_sigma=config.temporal_sigma,
            )

            if anchor_z is None:
                anchor_z, _ = _estimate_anchor_z(gx_vol, gy_vol, gx_f, gy_f)
                diff_hwcz = (
                    volume_hwcz[:, :, :, anchor_z][:, :, :, None] - volume_hwcz
                ).astype(np.float32)
                diff_hwcz[:, :, :, anchor_z] = 0.0

            z_hat_hwt = _estimate_z_patchwise(
                gx_vol,
                gy_vol,
                gx_f,
                gy_f,
                anchor_z=anchor_z,
                win_half=config.win_half,
                patch_size=config.patch_size,
                overlap=config.overlap,
                tau_scale=config.parabolic_tau_scale,
                z_smooth_sigma_spatial=config.z_smooth_sigma_spatial,
                z_smooth_sigma_temporal=config.z_smooth_sigma_temporal,
                parallelization=config.parallelization,
                n_jobs=config.n_jobs,
            )

            # Keep only the core frames; halo context frames are written by
            # the batches that own them.
            core = slice(n_left, n_left + n_core)
            z_hat_hwt = z_hat_hwt[:, :, core]

            # Persist z-shifts in MATLAB-style 1-based slice coordinates.
            z_batch_thwc = _from_hwct(
                (z_hat_hwt + 1.0)[:, :, None, :].astype(np.float32)
            )
            z_writer.write_frames(z_batch_thwc)

            if corrected_writer is not None and diff_hwcz is not None:
                corrected_hwct = _apply_z_correction(
                    batch_hwct[:, :, :, core], z_hat_hwt, diff_hwcz
                )
                corrected_thwc = _from_hwct(corrected_hwct)
                corrected_writer.write_frames(
                    _clip_and_cast(corrected_thwc, config.output_dtype)
                )

            n_batches += 1
            print(f"Stage 2: processed batch {n_batches}")

    finally:
        input_reader.close()
        z_writer.close()
        if corrected_writer is not None:
            corrected_writer.close()

    if anchor_z is None:
        raise RuntimeError(
            "Stage 2 processed zero batches; no z-shift estimate produced"
        )

    np.savez(
        str(output_root / "stage2_metadata.npz"),
        anchor_z_0based=np.array(anchor_z, dtype=np.int32),
        anchor_z_1based=np.array(anchor_z + 1, dtype=np.int32),
        volume_path=str(volume_path),
        input_path=str(input_path),
        prealigned_recording_path=(
            "" if prealigned_recording_path is None else str(prealigned_recording_path)
        ),
    )

    # Re-load before saving: run_recording_prealignment updated status.json
    # after our snapshot was taken, and saving the stale dict would erase its
    # "recording_prealign" completion marker.
    status = load_or_create_status(output_root)
    status["stage2"] = "done"
    status["anchor_z"] = int(anchor_z)
    status["anchor_z_1based"] = int(anchor_z + 1)
    save_status(output_root, status)

    elapsed = time() - start_time
    print(f"Stage 2 complete in {elapsed:.2f}s")
    return {
        "z_shift_path": z_shift_path,
        "corrected_path": corrected_path if config.write_corrected else None,
        "anchor_z": anchor_z,
        "prealigned_recording_path": prealigned_recording_path,
    }


def run_stage3(
    config: ZAlignConfig,
    volume_path: Optional[Path] = None,
    z_shift_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Stage 3: simulate a z-shift-only recording from volume + z-shift.

    Returns
    -------
    Path or None
        Simulated output path (or None if simulation disabled).
    """
    if not config.write_simulated:
        print("Stage 3: simulation disabled by config (write_simulated=false)")
        return None

    start_time = time()
    output_root = config.resolve_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    status = load_or_create_status(output_root)

    simulated_path = config.resolve_simulated_output_file()
    simulated_path.parent.mkdir(parents=True, exist_ok=True)

    if config.resume and status.get("stage3") == "done" and simulated_path.exists():
        print(f"Stage 3: reusing existing simulation {simulated_path}")
        return simulated_path

    if volume_path is None:
        volume_path = config.resolve_reference_volume_path()
    if z_shift_path is None:
        z_shift_path = config.resolve_z_shift_file()

    if not volume_path.exists():
        raise FileNotFoundError(f"Reference volume not found: {volume_path}")
    if not z_shift_path.exists():
        raise FileNotFoundError(f"z_shift file not found: {z_shift_path}")

    volume_hwcz = _load_volume(config, volume_path)
    H, W, C, _ = volume_hwcz.shape

    z_reader = get_video_file_reader(
        str(z_shift_path),
        buffer_size=config.input_buffer_size,
        bin_size=1,
    )
    sim_fmt = _parse_output_format(simulated_path, fallback="TIFF")
    sim_writer = get_video_file_writer(str(simulated_path), sim_fmt)

    try:
        n_batches = 0
        while z_reader.has_batch():
            z_thwc = _ensure_thwc(z_reader.read_batch()).astype(np.float32, copy=False)
            if z_thwc.shape[1] != H or z_thwc.shape[2] != W:
                raise ValueError(
                    "z_shift dimensions do not match reference volume: "
                    f"z {(z_thwc.shape[1], z_thwc.shape[2])} vs volume {(H, W)}"
                )
            if z_thwc.shape[3] < 1:
                raise ValueError("z_shift batch must have at least one channel")

            # z_shift is stored as 1-based slice IDs for MATLAB parity.
            z_hwt = np.transpose(z_thwc[:, :, :, 0], (1, 2, 0)).astype(np.float64) - 1.0
            sim_hwct = _simulate_from_z(volume_hwcz, z_hwt)
            if sim_hwct.shape[2] != C:
                raise RuntimeError("Internal channel mismatch in simulated output")

            sim_thwc = _from_hwct(sim_hwct)
            sim_writer.write_frames(_clip_and_cast(sim_thwc, config.output_dtype))

            n_batches += 1
            print(f"Stage 3: processed batch {n_batches}")

    finally:
        z_reader.close()
        sim_writer.close()

    status["stage3"] = "done"
    save_status(output_root, status)

    elapsed = time() - start_time
    print(f"Stage 3 complete in {elapsed:.2f}s")
    return simulated_path


def run_all_stages(
    config: ZAlignConfig,
    of_options_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run all z-align stages.

    Returns
    -------
    dict
        Collected stage outputs.
    """
    print("=" * 60)
    print("Z-ALIGN STAGE 1: Build/Load Reference Volume")
    print("=" * 60)
    volume_path = run_stage1(config, of_options_override=of_options_override)

    print("\n" + "=" * 60)
    print("Z-ALIGN STAGE 2: Estimate z-shifts and Correct Signal")
    print("=" * 60)
    stage2_out = run_stage2(config, volume_path=volume_path)

    simulated_path = None
    if config.write_simulated:
        print("\n" + "=" * 60)
        print("Z-ALIGN STAGE 3: Simulate z-shift-only recording")
        print("=" * 60)
        simulated_path = run_stage3(
            config,
            volume_path=volume_path,
            z_shift_path=stage2_out["z_shift_path"],
        )
    else:
        print("\nSkipping Stage 3 (write_simulated=false)")

    return {
        "volume_path": volume_path,
        **stage2_out,
        "simulated_path": simulated_path,
    }
