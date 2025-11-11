# Session API Reference

The session module provides multi-recording session processing with MATLAB-compatible workflow for large-scale 2-photon microscopy experiments.

## Overview

The session processing pipeline consists of three stages:

1. **Stage 1**: Per-recording motion correction with valid mask persistence
2. **Stage 2**: Inter-sequence displacement computation using phase cross-correlation and optical flow refinement
3. **Stage 3**: Valid mask alignment and final session mask computation

This implementation provides full parity with MATLAB `align_full_v3_checkpoint.m` and `get_session_valid_index_v3_progressprint.m`.

## Configuration

### SessionConfig

```python
from pyflowreg.session.config import SessionConfig
```

::: pyflowreg.session.config.SessionConfig
    options:
        members:
            - root
            - pattern
            - center
            - output_root
            - final_results
            - resume
            - scheduler
            - flow_backend
            - backend_params
            - cc_upsample
            - sigma_smooth
            - alpha_between
            - iterations_between

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
center = "recording_03.tif"  # Optional, auto-selects middle if not specified
output_root = "compensated_outputs"
final_results = "final_results"
resume = true
scheduler = "local"  # or "array" for HPC

# Flow backend configuration
flow_backend = "flowreg"  # or "torch", "jax", etc.
[backend_params]
device = "cuda:0"  # For torch backend

# Stage 2 parameters
cc_upsample = 4
sigma_smooth = 6.0
alpha_between = 25.0
iterations_between = 100
```

## Stage 1: Per-Recording Compensation

### run_stage1

```python
from pyflowreg.session.stage1_compensate import run_stage1
```

::: pyflowreg.session.stage1_compensate.run_stage1
    options:
        show_source: false

**Features:**
- Automatic input file discovery
- Resume support with HDF5 completeness verification
- Temporal average computation with streaming (memory-efficient)
- Valid mask persistence via `idx.hdf`
- Atomic file writes for crash safety

**Example:**
```python
from pyflowreg.session import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1

config = SessionConfig.from_toml("session.toml")

# Override OFOptions directly on the config (or set via TOML/YAML)
config.flow_options = {
    "quality_setting": "fast",
    "save_w": True,
    "buffer_size": 1000,
}

output_folders = run_stage1(config)
```

### run_stage1_array

```python
from pyflowreg.session.stage1_compensate import run_stage1_array
```

::: pyflowreg.session.stage1_compensate.run_stage1_array
    options:
        show_source: false

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

::: pyflowreg.session.stage2_between_avgs.run_stage2
    options:
        show_source: false

**Algorithm:**
1. Load temporal averages from Stage 1
2. Identify center reference (auto or specified)
3. For each non-center recording:
   - Phase cross-correlation initialization (subpixel)
   - Gaussian smoothing (σ=6)
   - Optical flow refinement (α=25, iterations=100)
4. Save displacement fields as `w_to_reference.npz`

**Backend Selection:**

Stage 2 respects the `flow_backend` setting in configuration:
```python
config = SessionConfig(
    root="/data/",
    flow_backend="torch",  # Use PyTorch backend
    backend_params={"device": "cuda:0"}
)
middle_idx, center_file, displacements = run_stage2(config)
```

## Stage 3: Valid Mask Computation

### run_stage3

```python
from pyflowreg.session.stage3_valid_mask import run_stage3
```

::: pyflowreg.session.stage3_valid_mask.run_stage3
    options:
        show_source: false

**Processing:**
1. Load per-frame valid masks from `idx.hdf`
2. Compute temporal AND for each sequence
3. Warp masks to reference frame using displacement fields
4. Compute final mask as intersection of all aligned masks
5. Save comprehensive results bundle

**Output Files:**
- `final_valid_idx.png`: Final session mask (visual)
- `final_valid_idx.npz`: Complete results (Python)
- `final_valid_idx.mat`: MATLAB-compatible output
- Per-sequence masks and aligned versions

**NPZ Bundle Contents:**
```python
import numpy as np

data = np.load("final_results/final_valid_idx.npz")
data.keys()
# ['final_valid', 'aligned_valid_masks', 'per_seq_valid_masks',
#  'displacement_fields_u', 'displacement_fields_v',
#  'temporal_averages', 'compensated_h5_paths',
#  'reference_average', 'middle_idx']
```

**Aligned video export:** Stage 3 reprojects each per-recording `compensated.hdf5` into the session reference grid via `align_sequence()`. Tune behavior with `SessionConfig.align_chunk_size` (batch size) and `SessionConfig.align_output_format` (e.g., `TIFF`, `HDF5`). Outputs land in `final_results/aligned_<recording>.<ext>` and are skipped on resume when `resume=True`.

## Command-Line Interface

### pyflowreg-session

The session module provides a comprehensive CLI:

```bash
# Run complete pipeline
pyflowreg-session session.toml

# Run specific stages
pyflowreg-session session.toml --stage 1
pyflowreg-session session.toml --stage 2
pyflowreg-session session.toml --stage 3

# Array job mode (auto-detects task ID)
pyflowreg-session session.toml --stage 1 --array-job

# Override parameters
pyflowreg-session session.toml --no-resume --stage 1
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

::: pyflowreg.core.warping.backward_valid_mask
    options:
        show_source: false

::: pyflowreg.core.warping.imregister_binary
    options:
        show_source: false

::: pyflowreg.core.warping.compute_batch_valid_masks
    options:
        show_source: false

## Helper Functions

### get_array_task_id

```python
from pyflowreg.session.config import get_array_task_id
```

::: pyflowreg.session.config.get_array_task_id
    options:
        show_source: false

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

## Complete Example

```python
from pyflowreg.session import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1
from pyflowreg.session.stage2_between_avgs import run_stage2
from pyflowreg.session.stage3_valid_mask import run_stage3

# Configure session
config = SessionConfig(
    root="/data/2p_experiment/",
    pattern="recording_*.tif",
    center="recording_005.tif",
    output_root="compensated",
    final_results="results",
    resume=True,
    flow_backend="flowreg",
    cc_upsample=4,
    sigma_smooth=6.0,
    alpha_between=25.0,
    iterations_between=100
)

# Stage 1: Motion correct each recording
print("Running Stage 1...")
config.flow_options = {
    "quality_setting": "balanced",
    "save_valid_idx": True,
    "save_w": False,
}
output_folders = run_stage1(config)

# Stage 2: Align temporal averages
print("Running Stage 2...")
middle_idx, center_file, displacement_fields = run_stage2(config)

# Stage 3: Compute final valid mask
print("Running Stage 3...")
final_mask = run_stage3(config, middle_idx, displacement_fields)

print(f"Final valid region: {np.sum(final_mask)} pixels")
```

## MATLAB Compatibility

The session module maintains full compatibility with MATLAB Flow-Registration:

- File formats match exactly (HDF5, NPZ, MAT)
- Numerical results match within floating-point precision
- Resume behavior identical (can mix MATLAB/Python stages)
- Array job indexing compatible (1-based conversion handled)

**Verified against:**
- `align_full_v3_checkpoint.m`
- `get_session_valid_index_v3_progressprint.m`
