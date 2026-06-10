# Configuration

PyFlowReg uses the `OFOptions` class for comprehensive configuration of all motion correction parameters.

## Quality Settings

Quality settings control the finest pyramid level computed, balancing speed and accuracy.

```{literalinclude} ../snippets/user_guide/configuration/quality_settings.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Pyramid Levels Explained

The optical flow solver operates on multi-scale image pyramids. Level 0 (L0) is the finest, full-resolution level; higher levels are coarser:

- **Level 0**: Full resolution, used by `quality_setting="quality"` (the default)
- **Level 4**: Coarser scale used by `quality_setting="balanced"`
- **Level 6**: Coarse preview scale used by `quality_setting="fast"`

Stopping at a higher minimum level skips the finest scales, reducing computation time at the cost of accuracy. Setting `min_level` to a non-negative value overrides the preset and switches `quality_setting` to `"custom"`; resetting `min_level` to `-1` restores the previous preset.

## Core Optical Flow Parameters

```{literalinclude} ../snippets/user_guide/configuration/core_parameters.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Alpha (Smoothness Weight)

Controls the tradeoff between fitting the data and enforcing smooth flow fields. `alpha` is a 2-tuple `(alpha_x, alpha_y)` weighting smoothness along the x and y directions; a scalar is expanded to `(alpha, alpha)`. The default is `(1.5, 1.5)`.

- **Lower alpha**: Fits the data more closely and captures rapid motion changes, but is more noise-sensitive
- **Higher alpha**: Smoother flow fields, may miss sharp motion boundaries

### Iterations

Number of SOR iterations at each pyramid level (default 50). More iterations improve convergence with diminishing returns; fewer iterations speed up previews but may not fully converge.

See the [Parameter Guide](../theory/parameters.md) for guidance on the diffusion parameters `a_smooth` and `a_data`.

### Data Term and GNC

`constancy_assumption` selects the data term of the variational solver:

- `"gc"` (alias `"gradient"`): gradient constancy, the default, matching the MATLAB Flow-Registration data term
- `"gray"` (alias `"brightness"`): gray-value constancy
- `"cs"` (alias `"census"`): census constancy

Aliases are normalized to the serialized values `"gc"`, `"gray"`, and `"cs"` during validation. These data terms are implemented by the `flowreg` flow backend; the `diso` flow backend raises a `ValueError` for anything other than `"gc"`.

Two optional fields enable graduated non-convexity (GNC):

- `gnc_schedule` (default `None`): stage weights interpolating from a quadratic (0.0) to a fully robust (1.0) penalty, e.g. `gnc_schedule=(0.0, 0.5, 1.0)`. The schedule must be a 1D sequence with at least two entries in [0, 1], monotone nondecreasing, starting at 0.0 and ending at 1.0. Each stage reruns the pyramid, warm-started from the previous stage's result. With the default `None`, the standard single-pass solver is used.
- `warping_steps` (default `None`): number of warp/relinearize steps per pyramid level in GNC mode; must be at least 1. When GNC is active and `warping_steps` is not set, 10 steps per level are used; the value is ignored when GNC is off.

The `diso` flow backend also rejects `gnc_schedule` and `warping_steps`. See [Data Terms](../theory/data_terms.md) for the full background.

## Preprocessing

### Temporal Binning

Bin frames temporally to improve SNR and reduce computation:

```{literalinclude} ../snippets/user_guide/configuration/temporal_binning.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

Temporal binning is applied by the video reader. The effective frame rate is reduced by the binning factor.

### Gaussian Filtering

Apply Gaussian filtering to reduce noise:

```{literalinclude} ../snippets/user_guide/configuration/gaussian_filtering.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

The default is `[[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]`. Stronger filtering helps with very noisy data but can reduce spatial accuracy; see the [Parameter Guide](../theory/parameters.md) for tradeoffs.

### Channel Normalization

Control channel normalization for multi-channel data:

```{literalinclude} ../snippets/user_guide/configuration/channel_normalization.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

Normalization modes:
- `"joint"`: Normalize all channels together using global statistics (default)
- `"separate"`: Normalize each channel independently

## Backend Selection

### GPU Support Installation

To use the GPU flow backends (`flowreg_torch`, `flowreg_cuda`), install PyFlowReg with the GPU extra:

```bash
pip install pyflowreg[gpu]
```

**Linux/Windows:** Installs both PyTorch and CuPy (`cupy-cuda12x`) for CUDA acceleration. Requirements:
- NVIDIA GPU with CUDA support
- CUDA 12.x installed
- Compatible GPU drivers

**macOS:** Installs only PyTorch (CuPy is not available on macOS).

Without the GPU extra, the CPU flow backends (`flowreg` and `diso`) are available.

PyFlowReg supports multiple flow backends for the optical flow computation, including CPU and GPU implementations.

### Available Flow Backends

```{literalinclude} ../snippets/user_guide/configuration/flow_backend.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

**Available flow backends:**

- **`flowreg`** (default): NumPy-based CPU implementation with a Numba-compiled level solver
  - Works on all systems without optional dependencies
  - Supports all executors

- **`flowreg_torch`**: variational solver with a PyTorch level solver, on CPU or GPU
  - Requires PyTorch
  - With `device=None`, uses CUDA if available, otherwise CPU

- **`flowreg_cuda`**: variational solver with a CuPy level solver on NVIDIA GPUs
  - Requires CuPy and CUDA

- **`diso`**: OpenCV `DISOpticalFlow` (Dense Inverse Search)
  - Patch-based alternative to the variational solver
  - Does not support the `"gray"`/`"cs"` data terms or GNC (see [Data Term and GNC](#data-term-and-gnc))

The GPU flow backends (`flowreg_torch`, `flowreg_cuda`) only support the sequential executor, and `diso` does not support multiprocessing; see [Parallelization](parallelization.md) for the executor compatibility rules and fallback behavior.

### GPU Backend Configuration

```python
from pyflowreg.motion_correction import compensate_recording, OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

options = OFOptions(
    input_file="video.h5",
    flow_backend="flowreg_torch",
    backend_params={"device": "cuda", "dtype": "float32"}  # or "cpu"
)

# GPU flow backends run with the sequential executor
config = RegistrationConfig(parallelization="sequential")
compensate_recording(options, config=config)
```

**Device selection** (`flowreg_torch`):
```python
# Automatic device selection (CUDA if available, otherwise CPU)
backend_params={"device": None}

# Force CUDA
backend_params={"device": "cuda"}

# Force CPU
backend_params={"device": "cpu"}

# Specific GPU
backend_params={"device": "cuda:1"}
```

The `flowreg_torch` backend additionally accepts `dtype` (`"float32"` or `"float64"`, default `"float64"`). The `flowreg_cuda` backend takes no device parameter and runs on the default CuPy device.

## Reference Selection

### Fixed Reference Frames

`reference_frames` accepts frame indices, an image file path, or a precomputed reference array. The default is `list(range(50, 500))`, i.e. frames 50-499:

```{literalinclude} ../snippets/user_guide/configuration/reference_frames.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

When a list of indices is given, the corresponding frames are read from the input, motion-compensated against their mean using increased regularization, and averaged to form the reference.

### Updating the Reference Frame

By default the reference stays fixed for the entire recording (`update_reference=False`). With `update_reference=True`, the batch pipeline (`compensate_recording`) re-estimates the preprocessed reference after each batch: up to the last 100 frames of the batch are warped onto the current reference using their computed displacement fields and averaged per channel.

```{literalinclude} ../snippets/user_guide/configuration/update_reference.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

Related fields:

- `update_initialization_w` (default `True`): propagates the flow initialization across batches; after each batch, the initialization is updated with the mean of the last (up to 20) flow fields.
- `n_references` (default `1`): number of references. Multi-reference mode is not fully implemented; values above 1 repeat a single computed reference and emit a warning.

## Cross-Correlation Pre-Alignment

For recordings with large translational offsets, an optional rigid pre-alignment based on phase cross-correlation can initialize the variational flow:

```{literalinclude} ../snippets/user_guide/configuration/cc_initialization.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

When enabled, the executors estimate a rigid (translation-only) shift between the reference and each frame by phase cross-correlation on images downsampled to at most `cc_hw`, apply it together with the current flow initialization, and let the variational solver refine only the remaining non-rigid residual. The rigid shift, initialization, and residual flow are summed into the final displacement field. See [Pre-Alignment](prealignment.md) for the full description.

## File I/O Configuration

### Input/Output Paths

```{literalinclude} ../snippets/user_guide/configuration/io_paths.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

Supported `output_format` values include `TIFF`, `HDF5`, `MAT`, the per-channel variants `MULTIFILE_TIFF`, `MULTIFILE_HDF5`, `MULTIFILE_MAT`, `CAIMAN_HDF5`, `SUITE2P_TIFF`, and the in-memory formats `ARRAY` and `NULL`; see [File Formats](file_formats.md). Note that `compensate_arr` returns arrays and ignores `output_format`; see the [Motion Correction API](../api/motion_correction.md).

### Saving Displacement Fields

```{literalinclude} ../snippets/user_guide/configuration/save_outputs.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Buffer Size

Control memory usage and I/O efficiency:

```{literalinclude} ../snippets/user_guide/configuration/buffer_size.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

Larger buffers improve I/O efficiency but increase memory usage.

## Validation

`OFOptions` uses Pydantic for automatic validation:

```{literalinclude} ../snippets/user_guide/configuration/validation.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## Saving and Loading Configuration

```{literalinclude} ../snippets/user_guide/configuration/save_load.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```
