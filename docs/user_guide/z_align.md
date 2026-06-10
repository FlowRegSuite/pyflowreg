# Z-Alignment Guide

This guide explains how to use the `pyflowreg.z_align` module to estimate and
correct depth (z) shifts in volumetric 2-photon recordings. It is a narrative
walkthrough of the pipeline and its configuration; for the class and function
signatures see the [z_align API reference](../api/z_align.md).

## The Problem: Depth Shift in Volumetric Recordings

The z-align module is modeled on the MATLAB patch-based z-shift prototype. The
core 2D motion correction in PyFlowReg corrects in-plane (xy) motion but does
not correct through-plane (z) motion. When the imaging plane drifts in depth
relative to a reference volume, the recorded frames sample different z slices of
the tissue over time.

The z-align pipeline addresses this by comparing a recording against a
compensated **reference volume** (a stack of z slices) and estimating, per
pixel and per frame, which z slice of the reference the recorded content best
matches. From these per-pixel z estimates it can write a z-corrected recording
and, optionally, a z-shift-only simulation reconstructed purely from the
reference volume.

## Pipeline Overview

The pipeline (`pyflowreg.z_align.pipeline`) runs in three stages, with an
optional recording-prealignment step. Stage completion is tracked in a
`status.json` file under the output root so runs can be resumed.

1. **Stage 1** (`run_stage1`): build or load a compensated reference volume.
2. **Stage 2** (`run_stage2`): estimate per-pixel z-shifts patch-wise and
   optionally write a z-corrected recording.
3. **Stage 3** (`run_stage3`): optionally simulate a z-shift-only recording from
   the reference volume and the estimated z-shifts.

`run_all_stages` runs Stage 1, then Stage 2, then (if `write_simulated` is
enabled) Stage 3.

### Recording Prealignment

`run_recording_prealignment` is an optional step that motion-compensates the
Stage 2 input recording in-plane *before* z-shift estimation, so the z search
is not confused by residual xy motion. It runs only when
`prealign_recording=True` (default `False`). It is also invoked automatically
from within `run_stage2`, so enabling the flag is sufficient; you do not have to
call it yourself.

When it runs, it:

- Resolves the recording to prealign from `input_file`.
- Builds a reference image by averaging leading frames of
  `reference_source_file`, or of `input_file` when `reference_source_file` is
  `None` (see [Reference image from a source recording](#reference-image-from-a-source-recording)).
- Calls `compensate_recording` with `alpha=stage1_alpha`,
  `quality_setting=stage1_quality_setting`, `buffer_size=input_buffer_size`,
  `bin_size=input_bin_size`, `update_reference=False`, plus the configured
  `flow_backend`/`backend_params` and any
  `recording_prealign_flow_options` overrides.
- Writes the prealigned recording to
  `<recording_prealigned_output_dir>/compensated.HDF5` and records
  `recording_prealign = "done"` in `status.json`.

When prealignment is active, Stage 2 reads the prealigned recording instead of
the raw input. Because prealignment already applied temporal binning,
Stage 2 then reads the prealigned recording **unbinned** (`bin_size=1`) to avoid
binning twice.

### Stage 1: Reference Volume

`run_stage1` produces the compensated reference volume used by Stages 2 and 3.
Its behavior depends on the configuration:

- If `reference_volume` is set, Stage 1 uses that existing volume file directly
  and marks the stage done without recomputing.
- Otherwise `volume_input_file` is required. If `prealign_stack=False`, the raw
  `volume_input_file` is used as the volume directly.
- If `prealign_stack=True` (default), Stage 1 runs `compensate_recording` on
  `volume_input_file` to build a compensated volume. The OFOptions are assembled
  from `stage1_alpha`, `stage1_quality_setting`, `stage1_bin_size`,
  `stage1_update_reference`, `flow_backend`, `backend_params`, and any
  `stage1_flow_options` overrides. The buffer size is `stage1_buffer_size`
  unless `stack_scans_per_slice` is set.

When `stack_scans_per_slice` is set, it is used both as the Stage 1 buffer size
and to force `update_reference=True`, so the recording is compensated one z
slice at a time with the reference adapting slice-by-slice. (This mirrors the
adaptive-reference z-stack strategy described in
[3D Z-Stack Alignment](3d_volumes.md).)

The compensated volume is written to `<volume_output_dir>/compensated.HDF5`, and
`status.json` records `stage1 = "done"` along with the resolved `volume_path`.

#### Reference image from a source recording

Before compensation, Stage 1 (and recording prealignment) can build a reference
*image* from a source recording, mirroring the MATLAB step
`reference = mean(reader.read_frames(1:N), 4)`. The leading
`reference_source_frames` (binned) frames of `reference_source_file` are read
with `reference_source_buffer_size`/`reference_source_bin_size` and averaged.
The averaged image is passed to `compensate_recording` as `reference_frames`.
If `reference_source_file` is `None`, no reference image is supplied for Stage 1
(compensation uses its own default reference); for recording prealignment it
falls back to averaging the input recording.

### Stage 2: Patch-Based z-Shift Estimation

`run_stage2` loads the reference volume into an `(H, W, C, Z)` array, then
processes the input recording in reader batches and estimates a per-pixel,
per-frame z map. The main steps as implemented:

1. **Volume gradients.** Each reference z slice is smoothed with a Gaussian
   (`spatial_sigma`) and its spatial gradients (gx, gy) are precomputed.

2. **Batched reading with temporal halo.** Input batches are read through a
   temporal-halo iterator. Each batch is extended with up to `halo` real context
   frames copied from the neighboring batches so that the temporal Gaussian
   filters see true neighbors at batch boundaries instead of reflected
   boundaries. The halo is
   `ceil(4*temporal_sigma) + ceil(4*z_smooth_sigma_temporal)` frames (the sum of
   the two temporal kernel radii, since SciPy truncates Gaussians at 4 sigma).
   This makes the filtered core frames equivalent to whole-recording processing,
   so z estimates do not depend on the reader buffer size. Context is limited to
   the immediately adjacent batch, so the halo should not exceed the buffer size
   (`input_buffer_size`). Only the core frames of each batch are written; halo
   frames are written by the batches that own them.

3. **Input gradients.** Each batch is smoothed spatially (`spatial_sigma`) and
   temporally (`temporal_sigma`), then differentiated to obtain per-frame
   gradients.

4. **Anchor slice.** On the **first batch only**, an anchor z index is estimated
   by summing the gradient-constancy error (sum of absolute gx and gy
   differences between volume slice and input) over all pixels and frames and
   taking the z slice with the smallest error. This anchor (0-based internally)
   is reused for all subsequent batches, and the per-slice difference volume used
   for correction is built relative to it.

5. **Patch scoring.** The frame is tiled into square patches of side
   `patch_size` with stride `round(patch_size * (1 - overlap))`. The z search is
   restricted to the window `[anchor_z - win_half, anchor_z + win_half]` clipped
   to the volume bounds. Each patch is scored against every candidate z slice
   using the same gradient-constancy error, and the best-scoring candidate is
   selected per frame.

6. **Parabolic subpixel refinement.** Around the best integer candidate, a
   parabola is fit to the three neighboring scores to obtain a sub-voxel offset
   in `[-0.5, 0.5]`. When the parabola is near-flat (curvature below
   `parabolic_tau_scale * max(|s0|, 1)`), the integer estimate is kept instead.

7. **Aggregation and smoothing.** Overlapping patch estimates are averaged per
   pixel (deterministic row-major accumulation), then the resulting z map is
   smoothed with a Gaussian over space (`z_smooth_sigma_spatial`) and time
   (`z_smooth_sigma_temporal`) and clipped to the search window.

The z map is written to `z_shift_file` (HDF5) in **1-based slice coordinates**
for MATLAB parity: the internal 0-based estimate has `1.0` added before writing.
If `write_corrected=True`, a z-corrected recording is also written by
interpolating the per-slice difference volume at the estimated z and adding it
to the input frames; integer outputs are clipped and rounded to `output_dtype`.

Stage 2 also writes `stage2_metadata.npz` (anchor z in both 0-based and 1-based
form, plus the resolved volume/input paths) and records `stage2 = "done"`,
`anchor_z`, and `anchor_z_1based` in `status.json`.

Patch scoring runs sequentially by default; set `parallelization="threading"`
(with `n_jobs`) to score patches on a thread pool. Accumulation is always done
in row-major order so results are deterministic regardless of the executor.

### Stage 3: Simulated z-Shift-Only Recording

`run_stage3` runs only when `write_simulated=True` (default). It reads the
z-shift file (subtracting `1.0` to return to 0-based slice coordinates) and, for
each frame, reconstructs the image by interpolating the reference volume along z
at the estimated per-pixel z. The result is written to `simulated_output_file`,
clipped and cast to `output_dtype`. This produces a recording that contains only
the z-shift-induced appearance changes, with in-plane content fixed to the
reference volume. On success `stage3 = "done"` is recorded.

## Configuration: `ZAlignConfig`

All pipeline parameters live in the `ZAlignConfig` Pydantic model
(`pyflowreg.z_align.config`). It can be built directly in Python or loaded from
a TOML/YAML file via `ZAlignConfig.from_toml()`, `from_yaml()`, or the
extension-dispatching `from_file()`. The model sets `extra="forbid"`, so unknown
keys raise a validation error rather than being silently ignored.

The fields below are grouped by purpose, with defaults as defined in
`config.py`.

### Core paths

| Field | Default | Purpose |
|-------|---------|---------|
| `root` | (required) | Base directory; must exist. Relative paths resolve against it. |
| `input_file` | (required) | Recording to estimate z-shifts for (Stage 2 input). |
| `volume_input_file` | `None` | Raw stack compensated in Stage 1. Required when `reference_volume` is not set. |
| `reference_volume` | `None` | Existing compensated volume; when set, Stage 1 skips the volume build. |
| `reference_source_file` | `None` | Recording whose leading frames are averaged into the Stage 1 / prealignment reference image. |

### Reference image building

| Field | Default | Purpose |
|-------|---------|---------|
| `reference_source_frames` | `2000` | Max number of (binned) frames averaged into the reference image. |
| `reference_source_buffer_size` | `10` | Reader batch size for the reference source. |
| `reference_source_bin_size` | `20` | Temporal binning applied when reading the reference source. |

### Output locations

| Field | Default | Purpose |
|-------|---------|---------|
| `output_root` | `z_align_outputs` | Directory for all outputs and `status.json` (resolved under `root`). |
| `volume_output_dir` | `aligned_stack` | Where Stage 1 writes the compensated volume (under `output_root`). |
| `recording_prealigned_output_dir` | `prealigned_recording` | Where the optional prealigned recording is written. |
| `z_shift_file` | `z_shift.HDF5` | Stage 2 per-pixel z-shift output. Must have an HDF5 extension (validated). |
| `corrected_output_file` | `compensated_shift_corrected.tif` | Stage 2 z-corrected recording. |
| `simulated_output_file` | `simulated_from_z.tif` | Stage 3 simulated recording. |

### Control flags

| Field | Default | Purpose |
|-------|---------|---------|
| `resume` | `True` | Reuse completed stage outputs recorded in `status.json`. |
| `prealign_stack` | `True` | Motion-compensate the raw stack in Stage 1 (vs. using it raw). |
| `prealign_recording` | `False` | Motion-compensate the input recording before Stage 2. |
| `write_corrected` | `True` | Write the z-corrected recording in Stage 2. |
| `write_simulated` | `True` | Run Stage 3 and write the simulated recording. |

### Stage 1 (volume build)

| Field | Default | Purpose |
|-------|---------|---------|
| `stage1_alpha` | `5.0` | OFOptions `alpha` for Stage 1 (and recording prealignment). |
| `stage1_quality_setting` | `quality` | OFOptions `quality_setting` for Stage 1 (and prealignment). |
| `stage1_buffer_size` | `500` | Reader buffer size for Stage 1 (superseded by `stack_scans_per_slice`). |
| `stage1_bin_size` | `1` | Temporal bin size for Stage 1 compensation. |
| `stage1_update_reference` | `True` | OFOptions `update_reference`; forced `True` when `stack_scans_per_slice` is set. |
| `stack_scans_per_slice` | `None` | Repeated scans per z slice; sets the Stage 1 buffer/bin size for reading slices. |
| `flow_backend` | `flowreg` | Optical-flow backend for Stage 1 and prealignment. |
| `backend_params` | `{}` | Backend-specific parameters passed to OFOptions. |
| `stage1_flow_options` | `None` | Extra OFOptions overrides for Stage 1 (inline mapping or path to a saved OF_options JSON). |
| `recording_prealign_flow_options` | `None` | Same, but for the optional recording prealignment. |

For `stage1_flow_options` and `recording_prealign_flow_options`, workflow-owned
I/O routing fields (`input_file`, `output_path`, `output_format`,
`output_file_name`, `naming_convention`) are stripped before the overrides are
applied; `reference_frames` is additionally stripped for Stage 1.

### Stage 2 (patch-based z estimation)

| Field | Default | Purpose |
|-------|---------|---------|
| `input_buffer_size` | `50` | Reader batch size for the Stage 2 input (and Stage 3 z-shift reading). |
| `input_bin_size` | `1` | Temporal binning for the input recording (skipped when reading a prealigned recording). |
| `volume_buffer_size` | `500` | Reader batch size when loading the reference volume. |
| `volume_bin_size` | `1` | Temporal binning when reading the volume as z slices (superseded by `stack_scans_per_slice`). |
| `win_half` | `10` | Half-width of the z search window around the anchor slice. |
| `patch_size` | `128` | Side length of the square spatial patches. |
| `overlap` | `0.75` | Fractional patch overlap; stride is `round(patch_size * (1 - overlap))`. Must satisfy `0 <= overlap < 1`. |
| `spatial_sigma` | `1.5` | Gaussian sigma for spatial smoothing before gradient computation. |
| `temporal_sigma` | `1.5` | Temporal Gaussian sigma applied to input frames before gradients. |
| `z_smooth_sigma_spatial` | `5.0` | Spatial Gaussian sigma for smoothing the z map. |
| `z_smooth_sigma_temporal` | `1.5` | Temporal Gaussian sigma for smoothing the z map. |
| `parabolic_tau_scale` | `1e-3` | Curvature threshold scale for sub-voxel refinement; near-flat parabolas keep the integer z. |
| `output_dtype` | `uint16` | NumPy dtype for corrected and simulated outputs; integer outputs are clipped and rounded. |
| `n_jobs` | `-1` | Worker count for patch scoring; `-1` uses all CPU cores. |
| `parallelization` | `sequential` | Patch-scoring mode: `sequential` or `threading`. |

Validation enforced by the field validators includes: `root` must exist and be a
directory; the integer batch/window fields must be >= 1; `stage1_alpha`, the
sigma fields, and `parabolic_tau_scale` must be > 0; `overlap` must lie in
`[0, 1)`; `z_shift_file` must end in `.h5`/`.hdf5`/`.hdf`; `output_dtype` must be
a valid NumPy dtype name; `n_jobs` must be `-1` or >= 1; and `parallelization`
must be `sequential` or `threading`.

### TOML example

The keys below match the `ZAlignConfig` field names exactly. `root` and
`input_file` are required; provide either `volume_input_file` (to build the
volume in Stage 1) or `reference_volume` (to reuse an existing volume).

```toml
# Core paths (root and input_file are required)
root = "/path/to/experiment"
input_file = "recording.tif"
volume_input_file = "reference_stack.tif"
# reference_volume = "aligned_stack/compensated.HDF5"  # alternative to volume_input_file

# Reference image built from leading frames of a source recording
reference_source_file = "reference_stack.tif"
reference_source_frames = 2000
reference_source_bin_size = 20

# Output locations (resolved under output_root, which is under root)
output_root = "z_align_outputs"
volume_output_dir = "aligned_stack"
z_shift_file = "z_shift.HDF5"
corrected_output_file = "compensated_shift_corrected.tif"
simulated_output_file = "simulated_from_z.tif"

# Control flags
resume = true
prealign_stack = true
prealign_recording = false
write_corrected = true
write_simulated = true

# Stage 1 (volume build)
stage1_alpha = 5.0
stage1_quality_setting = "quality"
flow_backend = "flowreg"
# stack_scans_per_slice = 10  # set when the stack has repeated scans per z slice

# Stage 2 (patch-based z estimation)
win_half = 10
patch_size = 128
overlap = 0.75
spatial_sigma = 1.5
temporal_sigma = 1.5
z_smooth_sigma_spatial = 5.0
z_smooth_sigma_temporal = 1.5
output_dtype = "uint16"
parallelization = "sequential"
n_jobs = -1
```

## Resume Semantics

Each stage records its completion in `output_root/status.json`. The relevant
keys are `stage1`, `recording_prealign`, `stage2`, and `stage3`, each set to
`"done"` when the stage finishes, alongside resolved paths and (for Stage 2) the
anchor z. The file is written atomically (written to a temporary file, then
renamed).

When `resume=True` (default), a stage is skipped only when both its
`status.json` marker is `"done"` **and** the expected output files exist on
disk:

- **Stage 1** reuses the existing volume when `stage1 == "done"` and the
  compensated volume file exists. (When `reference_volume` is set, or
  `prealign_stack=False`, no build is needed and the stage is marked done
  immediately.)
- **Recording prealignment** reuses the prealigned recording when
  `recording_prealign == "done"` and the file exists.
- **Stage 2** is skipped when `stage2 == "done"`, the z-shift file exists, and
  (if `write_corrected`) the corrected file exists.
- **Stage 3** is skipped when `stage3 == "done"` and the simulated file exists.

Setting `resume=False` forces every stage to recompute regardless of existing
markers.

## Command-Line Interface

The package installs a `pyflowreg-z-align` console script with a single `run`
subcommand:

```bash
# Run the full z-align workflow
pyflowreg-z-align run --config z_align.toml

# Run only one stage (1, 2, or 3)
pyflowreg-z-align run --config z_align.toml --stage 2

# Override stage-1 OFOptions from the CLI
pyflowreg-z-align run --config z_align.toml --of-params alpha=8 quality_setting=balanced
```

The `run` subcommand accepts:

| Argument | Description |
|----------|-------------|
| `--config`, `-c` (required) | Path to the config file (`.toml`/`.yaml`/`.yml`). |
| `--stage`, `-s` | Run only one stage (`1`, `2`, or `3`); default runs all applicable stages. |
| `--of-params KEY=VALUE ...` | Override stage-1 OFOptions; ignored (with a warning) when `--stage` is `2` or `3`. |

Override values are parsed as booleans (`true`/`false`), integers, floats, or
JSON lists/dicts where possible; anything else is passed through as a string.

## Output File Map

With the default configuration and `root`/`output_root` resolved as above, a
full run produces:

| Path (under `output_root`) | Stage | Notes |
|----------------------------|-------|-------|
| `status.json` | all | Stage completion markers and resolved paths. |
| `aligned_stack/compensated.HDF5` | Stage 1 | Compensated reference volume (omitted when `reference_volume` is provided or `prealign_stack=False`). |
| `prealigned_recording/compensated.HDF5` | prealignment | Only when `prealign_recording=True`. |
| `z_shift.HDF5` | Stage 2 | Per-pixel z map in 1-based slice coordinates; single channel, written via the HDF5 writer (default dataset `ch1`, MATLAB `(H, W, T)` layout). |
| `compensated_shift_corrected.tif` | Stage 2 | z-corrected recording (only when `write_corrected=True`). |
| `stage2_metadata.npz` | Stage 2 | Anchor z (0-based and 1-based) and resolved volume/input paths. |
| `simulated_from_z.tif` | Stage 3 | Simulated z-shift-only recording (only when `write_simulated=True`). |

The directory and file names above are the defaults; each is configurable
through the corresponding `ZAlignConfig` field. Output formats for the corrected
and simulated recordings follow the file extension you give (`.tif`/`.tiff` →
TIFF, `.h5`/`.hdf5`/`.hdf` → HDF5, `.mat` → MAT).

## See Also

- [z_align API reference](../api/z_align.md) — class and function signatures.
- [3D Z-Stack Alignment](3d_volumes.md) — the 2D adaptive-reference z-stack
  workflow that can produce a reference volume.
- [Configuration](configuration.md) — OFOptions parameter tuning.
