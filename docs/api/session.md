# Session API Reference

The session module provides multi-recording session processing for large-scale 2-photon microscopy experiments.

## Overview

The session processing pipeline consists of three stages:

1. **Stage 1**: Per-recording motion correction with valid mask persistence
2. **Stage 2**: Inter-sequence displacement computation using phase cross-correlation and optical flow refinement
3. **Stage 3**: Valid mask alignment and final session mask computation

The implementation is modeled on the MATLAB session workflow scripts `align_full_v3_checkpoint.m` and `get_session_valid_index_v3_progressprint.m`.

## Configuration

### SessionConfig

```python
from pyflowreg.session.config import SessionConfig
```

```{eval-rst}
.. autoclass:: pyflowreg.session.config.SessionConfig
   :members:
   :exclude-members: model_config, model_fields, model_computed_fields
```

**Configuration File Support**

Load from TOML:
```python
config = SessionConfig.from_toml("session.toml")
```

Load from YAML:
```python
config = SessionConfig.from_yaml("session.yml")
```

Auto-detect format:
```python
config = SessionConfig.from_file("session.toml")  # or .yml/.yaml
```

**Example TOML Configuration**

```toml
# session.toml
root = "/data/experiment/"
pattern = "*.tif"
center = "recording_03.tif"  # Optional, defaults to lexicographic middle file
output_root = "compensated_outputs"
final_results = "final_results"
resume = true
scheduler = "local"  # "local", "array" (HPC job array), or "dask"
n_workers = -1       # Stage 1 workers (-1 = all CPU cores)

# Flow backend configuration
flow_backend = "flowreg"  # or "flowreg_torch", "flowreg_cuda", "diso"

# Stage 1 parameters
stage1_quality_setting = "balanced"  # Optional OFOptions quality preset

# Stage 2 parameters
cc_upsample = 4
sigma_smooth = 6.0
alpha_between = 25.0
iterations_between = 100
stage2_constancy_assumption = "gc"  # Options: "gc", "gray", "cs"

# Stage 3 parameters
align_chunk_size = 64
align_output_format = "TIFF"

# Backend parameters (TOML table; must come after all top-level keys)
[backend_params]
device = "cuda:0"  # For flowreg_torch backend
```

## Stage 1: Per-Recording Compensation

### run_stage1

```python
from pyflowreg.session.stage1_compensate import run_stage1
```

```{eval-rst}
.. autofunction:: pyflowreg.session.stage1_compensate.run_stage1
```

**Features:**
- Automatic input file discovery
- Resume support with HDF5 completeness verification
- Temporal average computation with streaming (memory-efficient)
- Valid mask persistence via `idx.hdf`
- Atomic file writes for crash safety

**Example:**
```{literalinclude} ../snippets/api/session/run_stage1_example.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### run_stage1_array

```python
from pyflowreg.session.stage1_compensate import run_stage1_array
```

```{eval-rst}
.. autofunction:: pyflowreg.session.stage1_compensate.run_stage1_array
```

**Array Job Support:**

Auto-detects task ID from environment variables:
- `SLURM_ARRAY_TASK_ID` (SLURM)
- `SGE_TASK_ID` (Sun Grid Engine)
- `PBS_ARRAY_INDEX` (PBS/Torque)

**SLURM Example:**
```bash
#!/bin/bash
#SBATCH --array=1-10
#SBATCH --job-name=stage1

python -c "
from pyflowreg.session import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1_array

config = SessionConfig.from_toml('session.toml')
run_stage1_array(config)
"
```

## Stage 2: Inter-Sequence Alignment

### run_stage2

```python
from pyflowreg.session.stage2_between_avgs import run_stage2
```

```{eval-rst}
.. autofunction:: pyflowreg.session.stage2_between_avgs.run_stage2
```

**Algorithm:**
1. Load temporal averages from Stage 1
2. Identify center reference (auto or specified); the center recording gets a zero displacement field
3. For each non-center recording:
   - Gaussian smoothing of both averages (`sigma_smooth`, default 6.0) and normalization to [0, 1]
   - Rigid pre-alignment via subpixel phase cross-correlation (`cc_upsample`, default 4), applied to the current average as a constant-shift warp
   - Optical flow refinement (`alpha_between`, default 25.0; `iterations_between`, default 100)
   - Total displacement = refined flow + rigid initialization
4. Save displacement fields as `w_to_reference.npz` (arrays `u` and `v`) in each output folder

**Backend Selection:**

Stage 2 respects the `flow_backend` setting in configuration:
```python
config = SessionConfig(
    root="/data/",
    flow_backend="flowreg_torch",  # Use PyTorch backend
    backend_params={"device": "cuda:0"}
)
middle_idx, center_file, displacements = run_stage2(config)
```

### Troubleshooting Stage 2 alignment

Stage 2 alignment is controlled by the following `SessionConfig` fields:

- `sigma_smooth`: Gaussian smoothing sigma applied to both temporal averages before flow estimation. Increase for noisy averages; decrease if fine structure is being smoothed away.
- `cc_upsample`: Upsampling factor for the phase cross-correlation pre-alignment (1 = integer-pixel shifts). The rigid estimate is computed on images downsampled to at most 256x256 pixels and rescaled to the full grid.
- `alpha_between`: Regularization weight for the inter-sequence optical flow (applied isotropically as `(alpha, alpha)`). Larger values produce smoother, more rigid displacement fields.
- `iterations_between`: Number of solver iterations for the inter-sequence flow refinement.
- `stage2_constancy_assumption`: Data term for the Stage 2 flow. Accepts `"gc"` (default, gradient constancy), `"gray"`, and `"cs"` (census), plus the aliases `"gradient"`, `"brightness"`, and `"census"`. The `"diso"` backend only supports `"gc"`; other values raise a `ValueError`.

When `resume=True`, existing `w_to_reference.npz` files are reused without recomputation. After changing any of the parameters above, delete the affected `w_to_reference.npz` files (or set `resume=False`) to force recomputation.

## Stage 3: Valid Mask Computation

### run_stage3

```python
from pyflowreg.session.stage3_valid_mask import run_stage3
```

```{eval-rst}
.. autofunction:: pyflowreg.session.stage3_valid_mask.run_stage3
```

**Processing:**
1. Load per-frame valid masks from `idx.hdf`
2. Compute temporal AND for each sequence
3. Warp masks to reference frame using displacement fields
4. Compute final mask as intersection of all aligned masks
5. Save comprehensive results bundle

**Output Files:**
- `final_valid_idx.png`: Final session mask (visual)
- `final_valid_idx.npz`: Complete results (Python)
- `final_valid_idx.mat`: MATLAB-compatible output (written when SciPy is available; `middle_idx` is stored 1-based)
- Per-sequence masks (`<recording>_valid_idx.png`) and aligned versions (`<recording>_valid_idx_aligned.png`)

**NPZ Bundle Contents:**
```python
import numpy as np

data = np.load("final_results/final_valid_idx.npz")
data.keys()
# ['final_valid', 'aligned_valid_masks', 'per_seq_valid_masks',
#  'displacement_fields_u', 'displacement_fields_v',
#  'temporal_averages', 'compensated_h5_paths',
#  'reference_average', 'middle_idx', 'aligned_video_paths']
```

**Aligned video export:** Stage 3 reprojects each per-recording `compensated.hdf5` into the session reference grid via `align_sequence()`. Tune behavior with `SessionConfig.align_chunk_size` (batch size) and `SessionConfig.align_output_format` (e.g., `TIFF`, `HDF5`). Outputs land in `final_results/aligned_<recording>.<ext>` and are skipped on resume when `resume=True`.

## Command-Line Interface

### pyflowreg-session

The session module installs a `pyflowreg-session` command:

```bash
# Run complete pipeline
pyflowreg-session run --config session.toml

# Run specific stages
pyflowreg-session run --config session.toml --stage 1
pyflowreg-session run --config session.toml --stage 2
pyflowreg-session run --config session.toml --stage 3

# Array job mode (auto-detects task ID)
pyflowreg-session run --config session.toml --stage 1 --array

# Process a single recording by index (0-based)
pyflowreg-session run --config session.toml --stage 1 --index 2

# Override OFOptions parameters for Stage 1
pyflowreg-session run --config session.toml --stage 1 --of-params quality_setting=fast

# Run Stage 1 in parallel via Dask (requires dask-jobqueue)
pyflowreg-session dask --config session.toml --scheduler slurm --jobs 10
```

**Help:**
```bash
pyflowreg-session --help
```

## Warping Utilities

### Core Functions

```python
from pyflowreg.core.warping import (
    backward_valid_mask,
    imregister_binary,
    compute_batch_valid_masks
)
```

```{eval-rst}
.. autofunction:: pyflowreg.core.warping.backward_valid_mask
```

```{eval-rst}
.. autofunction:: pyflowreg.core.warping.imregister_binary
```

```{eval-rst}
.. autofunction:: pyflowreg.core.warping.compute_batch_valid_masks
```

## Helper Functions

### get_array_task_id

```python
from pyflowreg.session.config import get_array_task_id
```

```{eval-rst}
.. autofunction:: pyflowreg.session.config.get_array_task_id
```

### atomic_save_npy / atomic_save_npz

```python
from pyflowreg.session.stage1_compensate import atomic_save_npy, atomic_save_npz
```

Crash-safe file writing with write-to-temp then atomic replace:

```python
# Safe numpy array save
atomic_save_npy(Path("data.npy"), array)

# Safe npz archive save
atomic_save_npz(Path("data.npz"), array1=arr1, array2=arr2)
```

```{eval-rst}
.. autofunction:: pyflowreg.session.stage1_compensate.atomic_save_npy
```

```{eval-rst}
.. autofunction:: pyflowreg.session.stage1_compensate.atomic_save_npz
```

## Complete Example

```{literalinclude} ../snippets/api/session/complete_session_pipeline.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## MATLAB Compatibility

The session module is modeled on the MATLAB Flow-Registration session scripts `align_full_v3_checkpoint.m` and `get_session_valid_index_v3_progressprint.m`, which are referenced throughout the implementation. MATLAB-oriented conventions in the Python code include:

- Center reference selection defaults to the lexicographic middle file (MATLAB `ceil(num_records/2)`)
- Per-frame valid masks are stored as `idx.hdf` in each output folder
- Stage 3 writes `final_valid_idx.mat` via `scipy.io.savemat` with `middle_idx` converted to 1-based indexing
- `run_stage1_array` converts 1-based scheduler task IDs (SLURM/SGE/PBS) to 0-based Python indices
