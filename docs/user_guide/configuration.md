# Configuration

PyFlowReg uses the `OFOptions` class for comprehensive configuration of all motion correction parameters.

## Quality Settings

Quality settings control the finest pyramid level computed, balancing speed and accuracy.

```python
from pyflowreg.motion_correction import OFOptions

# Fast preview (pyramid level 3)
options = OFOptions(quality_setting="fast")

# Balanced quality (pyramid level 1, recommended)
options = OFOptions(quality_setting="balanced")

# Maximum quality (pyramid level 0, full resolution)
options = OFOptions(quality_setting="quality")
```

### Pyramid Levels Explained

The optical flow solver operates on multi-scale image pyramids. Lower level numbers mean finer scales:

- **Level 0**: Full resolution - captures finest motion details but slowest
- **Level 1**: Half resolution - good balance of speed and accuracy
- **Level 3**: 1/8 resolution - fast preview but misses fine details

Computation time decreases exponentially with higher minimum levels, while accuracy decreases.

## Core Optical Flow Parameters

```python
options = OFOptions(
    # Smoothness regularization weight
    alpha=4,  # Higher = smoother flow fields

    # Solver iterations per pyramid level
    iterations=50,  # More iterations = better convergence

    # Pyramid configuration
    levels=50,  # Maximum pyramid levels (auto-computed from image size)
    eta=0.8,  # Downsampling factor between levels
    min_level=1,  # Finest level to compute (set by quality_setting)

    # Nonlinearity parameters
    epsilon_s=0.001,  # Smoothness nonlinearity threshold
    epsilon_d=0.001,  # Data term nonlinearity threshold
)
```

### Alpha (Smoothness Weight)

Controls the tradeoff between fitting the data and enforcing smooth flow fields:

- **Low alpha (1-2)**: Fits data more closely, captures rapid motion changes, more noise-sensitive
- **Medium alpha (3-5)**: Balanced (recommended for 2P microscopy)
- **High alpha (6-10)**: Very smooth flow, may miss sharp motion boundaries

### Iterations

Number of SOR iterations at each pyramid level:

- **25-50**: Typical range for most applications
- **100+**: Better convergence but diminishing returns
- **10-20**: Fast preview, may not fully converge

## Preprocessing

### Spatial Binning

Reduce spatial resolution before processing to improve SNR and speed:

```python
options = OFOptions(
    bin_size=2,  # 2x2 binning reduces resolution by half
)
```

Binning is applied before optical flow computation. Output resolution matches the binned resolution.

### Gaussian Filtering

Apply Gaussian blur to reduce noise:

```python
options = OFOptions(
    gaussian_filter=1.0,  # Sigma in pixels
)
```

Useful for very noisy data, but can reduce spatial accuracy.

### Normalization

Control channel normalization for multi-channel data:

```python
options = OFOptions(
    normalize_channels=True,  # Normalize each channel (default)
    normalization_mode="reference",  # Use reference frame statistics
)
```

Normalization modes:
- `"reference"`: Normalize using reference frame statistics (recommended)
- `"individual"`: Normalize each frame independently
- `"none"`: No normalization

## Backend Selection

### GPU Support Installation

To use GPU-accelerated backends, install PyFlowReg with GPU support:

```bash
pip install pyflowreg[gpu]
```

**Linux/Windows:** Installs both PyTorch and CuPy for CUDA acceleration. Requirements:
- NVIDIA GPU with CUDA support
- CUDA 12.x installed
- Compatible GPU drivers

**macOS:** Installs only PyTorch (CuPy not available on macOS). Enables `flowreg_torch` backend with MPS (Metal Performance Shaders) on Apple Silicon.

Without GPU support installed, only the CPU backend (`flowreg`) is available.

PyFlowReg supports multiple computational backends for optical flow calculation, including CPU and GPU acceleration.

### Available Backends

```python
options = OFOptions(
    flow_backend="flowreg",  # Choose backend
    backend_params={"device": "cuda"}  # Backend-specific parameters
)
```

**Available backends:**

- **`flowreg`** (default): NumPy-based CPU implementation with Numba JIT compilation
  - Best compatibility, works on all systems
  - Good performance for small to medium datasets
  - Supports all parallelization modes

- **`flowreg_torch`**: PyTorch-based implementation supporting CPU and GPU
  - Requires PyTorch installation
  - Automatically uses CUDA if available, falls back to CPU
  - Significantly faster on GPU for large datasets
  - **Requires sequential executor** (GPU memory management constraint)

- **`flowreg_cuda`**: CuPy-based GPU implementation
  - Requires CuPy and CUDA installation
  - Pure GPU computation for maximum performance
  - Best for very large datasets with NVIDIA GPUs
  - **Requires sequential executor** (GPU memory management constraint)

- **`diso`**: OpenCV DISOpticalFlow implementation
  - Designed for natural videos and non-2P microscopy data
  - Alternative dense inverse search algorithm

### GPU Backend Configuration

GPU backends require sequential execution due to GPU memory management:

```python
from pyflowreg.motion_correction import compensate_recording, OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

# PyTorch backend (automatic CPU/GPU selection)
options = OFOptions(
    input_file="video.h5",
    flow_backend="flowreg_torch",
    backend_params={"device": "cuda", "dtype": "float32"}  # or "cpu"
)

# Executor is automatically set to sequential for GPU backends
config = RegistrationConfig(parallelization="sequential")
compensate_recording(options, config=config)
```

**Device selection:**
```python
# Automatic device selection (CUDA if available, otherwise CPU)
backend_params={"device": None}

# Force CUDA
backend_params={"device": "cuda"}

# Force CPU (PyTorch only)
backend_params={"device": "cpu"}

# Specific GPU
backend_params={"device": "cuda:1"}
```

**Note:** If you request multiprocessing or threading with a GPU backend, PyFlowReg will automatically fall back to sequential execution and issue a warning.

### Performance Considerations

**For small datasets (< 1000 frames):**
```python
options = OFOptions(flow_backend="flowreg")  # CPU sufficient
```

**For large datasets with GPU:**
```python
options = OFOptions(
    flow_backend="flowreg_torch",
    backend_params={"device": "cuda", "dtype": "float32"}
)
```

**For maximum GPU performance:**
```python
options = OFOptions(
    flow_backend="flowreg_cuda",
    backend_params={"device": "cuda"}
)
```

## Reference Selection

### Fixed Reference Frames

Use specific frames as reference:

```python
# Single frame
options = OFOptions(reference_frames=[0])

# Average multiple frames
options = OFOptions(reference_frames=list(range(100, 200)))
```

### Automatic Reference Selection

Let PyFlowReg find stable frames:

```python
options = OFOptions(
    reference_frames=None,  # Auto-select
    auto_reference_method="variance",  # Use low-variance frames
)
```

### Reference Pre-Alignment

Enable cross-correlation pre-alignment for improved robustness:

```python
options = OFOptions(
    prealignment=True,  # Enable reference pre-alignment
    prealignment_method="xcorr",  # Cross-correlation
)
```

Pre-alignment performs initial rigid registration before optical flow, improving convergence for large displacements.

## File I/O Configuration

### Input/Output Paths

```python
options = OFOptions(
    input_file="raw_video.h5",
    output_path="results/",
    output_format="HDF5",  # HDF5, TIFF, or MAT
)
```

### Saving Displacement Fields

```python
options = OFOptions(
    save_w=True,  # Save displacement fields
    output_typename="float32",  # Data type for output
)
```

### Buffer Size

Control memory usage and I/O efficiency:

```python
options = OFOptions(
    buffer_size=100,  # Number of frames per batch
)
```

Larger buffers improve I/O efficiency but increase memory usage.

## Validation

`OFOptions` uses Pydantic for automatic validation:

```python
# Invalid parameter raises validation error
options = OFOptions(alpha=-1)  # ValueError: alpha must be positive

# Type conversion is automatic
options = OFOptions(alpha="4")  # Converts string to float
```

## Saving and Loading Configuration

```python
# Save configuration
options.save("config.json")

# Load configuration
options = OFOptions.load("config.json")

# Export to MATLAB-compatible format
options.save_matlab("config.mat")
```
