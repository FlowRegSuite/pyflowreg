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
