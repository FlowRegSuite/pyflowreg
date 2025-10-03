# Quickstart Guide

This guide will get you started with PyFlowReg using both array-based and file-based motion correction workflows.

## Basic Array-Based Workflow

The simplest way to use PyFlowReg is with in-memory arrays using `compensate_arr`:

```python
import numpy as np
from pyflowreg.motion_correction import compensate_arr, OFOptions

# Load your 2-photon video (T, H, W, C)
video = np.load("my_video.npy")

# Create reference from first 10 frames
reference = np.mean(video[:10], axis=0)

# Configure motion correction
options = OFOptions(quality_setting="balanced")

# Run compensation - returns registered video and flow fields
registered, flow = compensate_arr(video, reference, options)
```

### Quality Settings

PyFlowReg provides preset quality configurations that control the finest pyramid level computed:

- `quality_setting="fast"` - Computes to pyramid level 3, suitable for preview
- `quality_setting="balanced"` - Computes to pyramid level 1 (recommended)
- `quality_setting="quality"` - Computes to pyramid level 0 (full resolution), maximum accuracy

Finer pyramid levels capture smaller motion details but require more computation time.

## File-Based Workflow

For large datasets, use `compensate_recording` with file-based I/O:

```python
from pyflowreg.motion_correction import compensate_recording, OFOptions

# Configure with input/output paths
options = OFOptions(
    input_file="my_video.h5",
    output_path="results/",
    output_format="HDF5",
    quality_setting="balanced",
    reference_frames=list(range(100, 201)),  # Frames 100-200 as reference
    save_w=True  # Save displacement fields
)

# Run compensation (auto-selects parallelization backend)
compensate_recording(options)
```

## Parallel Processing

PyFlowReg provides three parallelization backends for batch processing:

- **Sequential** - Processes frames one-by-one, most memory-efficient
- **Threading** - Parallel processing using threads, good for I/O-bound operations
- **Multiprocessing** - Parallel processing using processes with shared memory, best for CPU-bound workloads (auto-selected by default if available)

By default, PyFlowReg auto-selects the best available backend (prefers multiprocessing, then threading, then sequential). To manually configure:

```python
from pyflowreg.motion_correction import compensate_recording, OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

options = OFOptions(
    input_file="my_video.h5",
    output_path="results/",
    quality_setting="balanced"
)

# Manual executor selection
config = RegistrationConfig(
    n_jobs=-1,                  # Use all CPU cores (-1) or specify number
    parallelization="threading" # "sequential", "threading", or "multiprocessing"
)

compensate_recording(options, config=config)
```

**Note:** A GPU implementation based on red-black Gauss-Seidel iteration is currently in development.

## Examples and Notebooks

The repository contains demo scripts under `examples/` and demo notebooks under `notebooks/`. The demos with the Jupiter sequence should run out of the box.

### Example Scripts

#### jupiter_demo.py
File-based motion compensation workflow demonstrating:
- TIFF to HDF5 conversion with multi-channel simulation
- Reference frame selection from stable video segment
- Parallel batch processing
- Metadata and displacement field saving

#### jupiter_demo_arr.py
Array-based motion compensation using `compensate_arr`:
- In-memory processing for smaller datasets
- Direct array manipulation and analysis
- Fast iteration for parameter tuning

#### jupiter_demo_live.py
Real-time motion correction with adaptive reference:
- Online processing with `FlowRegLive` class
- Adaptive reference frame updating
- Quality setting optimized for speed

#### synth_evaluation.py
Synthetic data evaluation and benchmarking:
- Ground truth comparison
- Quantitative accuracy assessment
- Performance profiling

### Jupyter Notebooks

#### jupiter_demo.ipynb
Comprehensive demonstration of optical flow-based motion compensation on Jupiter atmospheric distortion video, including:
- Multi-channel simulation by duplicating single-channel data
- HDF5 format conversion and multi-file handling
- Reference frame selection from stable segments (frames 100-200)
- Temporal slice analysis showing motion artifacts reduction
- Time course extraction at impact location with SNR improvement calculations
- Average frame comparison and motion blur quantification
- Side-by-side animations comparing original vs compensated videos
- Motion statistics visualization (displacement magnitude, divergence, translation)

#### flow_visualization.ipynb
Motion field visualization and quality control:
- Quiver plots of displacement fields
- Color-coded motion magnitude maps
- Divergence and curl analysis
- Quality metrics for registration assessment

### Running the Examples

Run example scripts as modules from the project root:

```bash
# Array-based demo
python -m examples.jupiter_demo_arr

# File-based demo
python -m examples.jupiter_demo

# Real-time demo
python -m examples.jupiter_demo_live
```

For notebooks:
```bash
jupyter notebook notebooks/jupiter_demo.ipynb
```

## Configuration Options

### Key Parameters

```python
options = OFOptions(
    # Flow parameters
    alpha=4,              # Smoothness regularization weight
    iterations=50,        # SOR iterations per pyramid level
    levels=50,            # Maximum pyramid levels
    eta=0.8,              # Pyramid downsampling factor
    min_level=3,          # Finest pyramid level to compute (quality_setting controls this)

    # Preprocessing
    bin_size=1,           # Spatial binning factor
    gaussian_filter=0,    # Gaussian filter sigma (0=disabled)

    # Reference
    reference_frames=[0, 1, 2, 3, 4],  # Frames to average for reference

    # I/O
    buffer_size=50,       # Frames to process per batch
    output_format="HDF5", # Output format (HDF5, TIFF, MAT)
    save_w=True,          # Save displacement fields
)
```

## Supported File Formats

PyFlowReg supports multiple file formats through its modular I/O system:

- **HDF5** (.h5, .hdf5) - Recommended for large datasets
- **TIFF** (.tif, .tiff) - Standard microscopy format
- **MAT** (.mat) - MATLAB compatibility
- **MDF** (.mdf) - Sutter MesaScope format (Windows only)

## Next Steps

- See the [User Guide](user_guide/index.md) for detailed workflows
- Check the [API Reference](api/index.md) for complete function documentation
- Read the [Theory](theory/index.md) section to understand the optical flow algorithm
