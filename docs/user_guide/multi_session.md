# Multi-Session Processing Guide

This guide covers processing multiple 2-photon microscopy recordings as a session, with inter-sequence alignment and valid mask computation.

## Overview

Multi-session processing is essential for longitudinal studies where you record from the same field of view across multiple time points or conditions. The session module provides:

- **Motion correction** of individual recordings
- **Cross-registration** between recordings
- **Valid mask computation** for consistent analysis regions
- **HPC support** for large-scale processing

## When to Use Session Processing

Use session processing when you have:

- Multiple recordings from the same field of view
- Longitudinal imaging sessions (days/weeks apart)
- Different experimental conditions on same neurons
- Need for pixel-perfect alignment across recordings

## Basic Workflow

### 1. Prepare Your Data

Organize recordings in a single directory:
```
experiment/
├── baseline_001.tif
├── baseline_002.tif
├── stimulus_001.tif
├── stimulus_002.tif
└── recovery_001.tif
```

### 2. Create Configuration

Create `session.toml`:
```toml
# Data location
root = "/path/to/experiment/"
pattern = "*.tif"

# Optional: specify center reference
# center = "baseline_002.tif"

# Output paths
output_root = "compensated_outputs"
final_results = "final_results"

# Processing options
resume = true
scheduler = "local"

# Optical flow backend
flow_backend = "flowreg"

# Stage 2 alignment parameters
cc_upsample = 4        # Subpixel accuracy
sigma_smooth = 6.0     # Gaussian smoothing
alpha_between = 25.0   # Regularization
iterations_between = 100
```

### 3. Run Processing

**Option A: Python Script**
```python
from pyflowreg.session import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1
from pyflowreg.session.stage2_between_avgs import run_stage2
from pyflowreg.session.stage3_valid_mask import run_stage3

# Load configuration
config = SessionConfig.from_toml("session.toml")

# Run all stages
output_folders = run_stage1(config)
middle_idx, center_file, displacements = run_stage2(config)
final_mask = run_stage3(config, middle_idx, displacements)
```

**Option B: Command Line**
```bash
# Run complete pipeline
pyflowreg-session session.toml

# Or run stages individually
pyflowreg-session session.toml --stage 1
pyflowreg-session session.toml --stage 2
pyflowreg-session session.toml --stage 3
```

## Deep Dive: Session Pipeline

The `pyflowreg.session` pipeline always runs the same three deterministic stages and records progress in `status.json` files so every step can resume safely. Knowing what each stage consumes and produces makes it easier to debug, scale to clusters, or mix MATLAB and Python tooling.

### Stage overview

| Stage | Inputs | Key operations | Primary outputs | Resume markers |
| --- | --- | --- | --- | --- |
| 1. Per-recording compensation | Recordings discovered with `root` + `pattern` | `discover_input_files()` → `compensate_recording()` with `save_valid_idx=True`; streaming temporal mean via `compute_and_save_temporal_average()`; HDF5 completeness checks | `compensated.hdf5`, `idx.hdf`, `temporal_average.npy`, per-recording `status.json` | `status.json["stage1"]`, verified `compensated.hdf5` frame count |
| 2. Inter-sequence alignment | Stage 1 temporal averages, center from `SessionConfig.resolve_center_file()` | `compute_between_displacement()` (phase cross-correlation init + optical flow refinement) | `w_to_reference.npz`, updated `status.json` | `status.json["stage2"]`, presence of `w_to_reference.npz` |
| 3. Valid mask & reprojection | Stage 1 masks, Stage 2 displacement fields | `load_idx_and_compute_mask()`, `imregister_binary()`, optional `align_sequence()` reprojection, final AND | `final_valid_idx.*`, aligned per-recording masks, optional `aligned_<recording>.<ext>` | `final_results/status.json["stage3"]`, `final_valid_idx.npz/png` |

### Stage 1 – Per-recording compensation

- `discover_input_files()` builds the sorted worklist before any processing begins (see `src/pyflowreg/session/stage1_compensate.py`).
- Each recording gets its own directory under `output_root/<stem>/` and runs through `compensate_recording()` with `save_valid_idx=True`, producing the motion-corrected video and `idx.hdf` mask stack.
- After compensation, the helper verifies that `compensated.hdf5` (case-insensitive) contains the expected frame count to guard against half-written files.
- Temporal averages are streamed with `compute_and_save_temporal_average()`, so even very long sequences never have to fit entirely in RAM.

**Outputs:** `compensated.hdf5`, `idx.hdf`, `temporal_average.npy`, `status.json` with `"stage1": "done"`, plus optional `statistics.npz`.

### Stage 2 – Inter-sequence alignment

- Temporal averages are reloaded from disk and the reference recording (center) is selected automatically or from `SessionConfig.center`.
- `compute_between_displacement()` smooths both averages, applies phase cross-correlation for a rigid guess, then refines with the configured flow backend (`src/pyflowreg/session/stage2_between_avgs.py`).
- Results are written to `w_to_reference.npz` (separate `u`/`v` arrays) so MATLAB users can load them directly.

**Outputs:** `w_to_reference.npz`, per-recording `status.json` updates, and `middle_idx` (0-based) pointing to the reference average.

### Stage 3 – Valid mask & aligned stacks

- `load_idx_and_compute_mask()` reduces each `idx.hdf` to a boolean mask via a temporal AND; raw and aligned masks are saved for inspection.
- Masks are warped into the session reference frame with `imregister_binary()`, mirroring MATLAB’s `get_session_valid_index_v3`.
- Each `compensated.hdf5` can optionally be reprojected into the reference grid via `align_sequence_video()` — controlled by `SessionConfig.align_chunk_size` and `SessionConfig.align_output_format`.
- The final mask is the logical AND of all aligned masks; the routine writes `.npz`, `.mat`, and `.png` variants and caches aligned videos as `aligned_<recording>.<ext>`.

**Outputs:** `final_valid_idx.{png,npz,mat}`, per-recording `*_valid_idx.png`/`*_valid_idx_aligned.png`, optional `aligned_<recording>.<ext>`, and `final_results/status.json`.

### Result artifacts at a glance

```
compensated_outputs/
└── recording_X/
    ├── compensated.hdf5
    ├── idx.hdf
    ├── temporal_average.npy
    ├── w_to_reference.npz
    └── status.json
final_results/
├── final_valid_idx.{png,npz,mat}
├── aligned_recording_X.tif  # format set by align_output_format
└── status.json
```

### Resume strategy

- Stage 1 re-runs automatically if `compensated.hdf5` is missing or its frame count no longer matches the source.
- Stage 2 skips work whenever `w_to_reference.npz` already exists.
- Stage 3 checks both `final_valid_idx.png` and `final_results/status.json` before recomputing.
- Because each artifact is deterministic, deleting a corrupt file and re-running just that stage is safe.

### CLI & scheduler coordination

- `pyflowreg-session run` orchestrates all three stages locally; `run_stage1_array()` lets HPC array jobs handle Stage 1 while Stages 2–3 run as follow-up jobs.
- The CLI prints reminders about running Stages 2–3 once all array tasks finish, and you can always call the Python APIs directly.
- For Dask, `pyflowreg-session dask` spins up a jobqueue cluster, parallelizes Stage 1, then executes Stages 2–3 serially on the client.

## Advanced Configuration

### Quality vs Speed Trade-offs

```toml
[flow_options]
quality_setting = "fast"     # Options: fast, balanced, quality
buffer_size = 1000           # Frames per batch
save_w = false               # Don't save displacement fields
save_valid_idx = true        # Required for Stage 3
```

Alternatively, point to a saved MATLAB/Python options file:

```toml
flow_options = "./saved_options/session_stage1.json"
```

### GPU Acceleration

Use PyTorch backend with CUDA:
```toml
flow_backend = "torch"
[backend_params]
device = "cuda:0"
```

### Custom Center Reference

By default, the lexicographic middle file is used as reference. Override with:
```toml
center = "specific_recording.tif"
```

## HPC / Cluster Processing

### SLURM Array Jobs

For large datasets, process Stage 1 in parallel:

**submit_stage1.sh:**
```bash
#!/bin/bash
#SBATCH --job-name=session_stage1
#SBATCH --array=1-20%5  # 20 recordings, max 5 parallel
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.9

python -c "
from pyflowreg.session import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1_array

config = SessionConfig.from_toml('session.toml')
config.scheduler = 'array'
run_stage1_array(config)
"
```

**submit_stages23.sh:**
```bash
#!/bin/bash
#SBATCH --job-name=session_stages23
#SBATCH --dependency=afterok:${STAGE1_JOB_ID}
#SBATCH --time=1:00:00
#SBATCH --mem=32G

module load python/3.9

python -c "
from pyflowreg.session import SessionConfig
from pyflowreg.session.stage2_between_avgs import run_stage2
from pyflowreg.session.stage3_valid_mask import run_stage3

config = SessionConfig.from_toml('session.toml')
middle_idx, center_file, displacements = run_stage2(config)
final_mask = run_stage3(config, middle_idx, displacements)
"
```

Submit sequence:
```bash
STAGE1_JOB=$(sbatch submit_stage1.sh | awk '{print $4}')
sbatch --dependency=afterok:${STAGE1_JOB} submit_stages23.sh
```

### SGE/PBS Support

The session module auto-detects array task IDs from:
- `SLURM_ARRAY_TASK_ID`
- `SGE_TASK_ID`
- `PBS_ARRAY_INDEX`

## Understanding the Output

### Directory Structure

After processing:
```
experiment/
├── compensated_outputs/
│   ├── baseline_001/
│   │   ├── compensated.hdf5      # Motion-corrected video
│   │   ├── temporal_average.npy   # Mean image
│   │   ├── idx.hdf               # Per-frame valid masks
│   │   ├── w_to_reference.npz    # Displacement to reference
│   │   └── status.json           # Processing status
│   ├── baseline_002/
│   │   └── ...
│   └── ...
└── final_results/
    ├── final_valid_idx.png        # Visual mask
    ├── final_valid_idx.npz        # Complete results
    ├── final_valid_idx.mat        # MATLAB compatible
    ├── *_valid_idx.png            # Per-sequence masks
    └── *_valid_idx_aligned.png    # Aligned masks
```

### Loading Results

```python
import numpy as np
from pathlib import Path

# Load final mask
results = np.load("final_results/final_valid_idx.npz")
final_mask = results['final_valid']

# Access all data
print(f"Valid pixels: {np.sum(final_mask)}/{final_mask.size}")
print(f"Reference recording: {results['middle_idx']}")

# Load motion-corrected videos
from pyflowreg.util.io.factory import get_video_file_reader

reader = get_video_file_reader("compensated_outputs/baseline_001/compensated.hdf5")
video = reader.read_frames(list(range(reader.frame_count)))

# Apply mask to analysis
masked_video = video[:, final_mask]  # Shape: (T, n_valid_pixels)
```

## Troubleshooting

### Memory Issues

**Problem:** Stage 1 runs out of memory

**Solution:** Reduce buffer size:
```python
config.flow_options = {"buffer_size": 500}
run_stage1(config)
```

### Incomplete Files

**Problem:** Crashed job left incomplete HDF5

**Solution:** Session module auto-detects and reruns:
- Verifies frame count matches input
- Uses atomic writes for crash safety
- Resume enabled by default

### Poor Alignment

**Problem:** Recordings don't align well

**Solutions:**
1. Increase iterations:
   ```toml
   iterations_between = 200
   ```

2. Adjust regularization:
   ```toml
   alpha_between = 15.0  # Lower = less smooth
   ```

3. Manually select better reference:
   ```toml
   center = "clearest_recording.tif"
   ```

### Array Job Failures

**Problem:** Some array tasks fail

**Solution:** Resubmit only failed tasks:
```bash
# Check which completed
ls compensated_outputs/*/status.json | wc -l

# Rerun specific task
SLURM_ARRAY_TASK_ID=5 python -c "..."
```

## Best Practices

### 1. Start Small
Test on subset first:
```python
config.pattern = "*_001.tif"  # Test with first of each condition
```

### 2. Verify Stage 1
Check temporal averages before proceeding:
```python
import matplotlib.pyplot as plt

for folder in output_folders:
    avg = np.load(folder / "temporal_average.npy")
    plt.figure()
    plt.imshow(avg, cmap='gray')
    plt.title(folder.name)
```

### 3. Monitor Displacement Magnitudes
Large displacements indicate problems:
```python
for w in displacement_fields:
    magnitude = np.sqrt(w[..., 0]**2 + w[..., 1]**2)
    print(f"Max displacement: {np.max(magnitude):.1f} pixels")
```

### 4. Save Intermediate Results
Enable for debugging:
```toml
[flow_options]
save_w = true           # Save displacement fields
save_meta_info = true   # Save statistics
```

## Integration with Analysis

### CaImAn Integration
```python
import caiman as cm

# Load using final mask
imgs = cm.load("compensated_outputs/*/compensated.hdf5")
imgs = imgs[:, final_mask]

# Run CNMF with consistent ROIs across sessions
cnm = cm.source_extraction.cnmf.CNMF(...)
cnm.fit(imgs)
```

### Suite2P Integration
```python
# Export masked videos for Suite2P
for h5_path in results['compensated_h5_paths']:
    video = load_video(h5_path)
    masked = video[:, final_mask]
    save_for_suite2p(masked)
```

## Performance Optimization

### Multi-threading Control
Prevent thread oversubscription in parallel processing:
```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
```

### Batch Size Tuning
Larger batches = better performance but more memory:
```python
# For 16GB RAM
config.flow_options = {"buffer_size": 1000}

# For 64GB RAM
config.flow_options = {"buffer_size": 5000}
```

### Storage Considerations
- Use local scratch for temporary files
- Output to parallel filesystem (Lustre/GPFS)
- Enable compression for final outputs:
  ```python
  config.flow_options = {"compression": "gzip"}
  ```

## MATLAB Interoperability

Results are fully compatible with MATLAB Flow-Registration:

```matlab
% Load Python results in MATLAB
load('final_results/final_valid_idx.mat');

% Use with MATLAB analysis
masked_pixels = video(final_valid);
```

Mix processing stages:
```bash
# Stage 1 in MATLAB
matlab -batch "align_full_v3_checkpoint('session.toml')"

# Stages 2-3 in Python
pyflowreg-session session.toml --stage 2
pyflowreg-session session.toml --stage 3
```
