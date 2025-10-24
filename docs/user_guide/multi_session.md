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

## Advanced Configuration

### Quality vs Speed Trade-offs

```toml
[of_options_override]
quality_setting = "fast"     # Options: fast, balanced, quality
buffer_size = 1000          # Frames per batch
save_w = false              # Don't save displacement fields
save_valid_idx = true       # Required for Stage 3
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
run_stage1(config, of_options_override={"buffer_size": 500})
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
[of_options_override]
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
of_options_override = {"buffer_size": 1000}

# For 64GB RAM
of_options_override = {"buffer_size": 5000}
```

### Storage Considerations
- Use local scratch for temporary files
- Output to parallel filesystem (Lustre/GPFS)
- Enable compression for final outputs:
  ```python
  of_options_override = {"compression": "gzip"}
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
