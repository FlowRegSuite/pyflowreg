# Parallelization

PyFlowReg provides multiple parallelization executors for batch processing, implemented through a runtime registry system. An *executor* is the mechanism that runs per-frame flow computation across the frames of a batch (sequential, threading, or multiprocessing). It is distinct from the *flow backend*, which is the optical flow computation implementation (`flowreg`, `diso`, `flowreg_torch`, `flowreg_cuda`). The two interact: each flow backend declares which executors it supports (see [Flow backends and executors](#flow-backends-and-executors)).

## Executors

### Sequential

**Single-threaded processing, most memory-efficient**

```{literalinclude} ../snippets/user_guide/parallelization/sequential_run.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

**When to use:**
- Debugging (easier to trace errors)
- Limited memory systems
- Small datasets

**Pros:**
- Lowest memory footprint
- Deterministic execution
- Simple error handling

**Cons:**
- Slowest processing
- No CPU parallelization

### Threading

**Parallel processing using Python threads**

```{literalinclude} ../snippets/user_guide/parallelization/threading_config.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

**When to use:**
- I/O-bound workloads
- Limited by disk/network speed
- Moderate CPU resources

**Pros:**
- Lower memory overhead than multiprocessing
- Good for I/O-bound operations
- Shared memory between threads

**Cons:**
- Limited by Python GIL for CPU-bound tasks
- Slower than multiprocessing for pure computation

### Multiprocessing

**Parallel processing using processes with shared memory (default)**

```{literalinclude} ../snippets/user_guide/parallelization/multiprocessing_config.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

**When to use:**
- CPU-bound workloads (recommended)
- Large multi-core systems
- Production processing

**Pros:**
- True parallel execution (bypasses GIL)
- Best performance for CPU-intensive tasks
- Shared memory for efficient data transfer

**Cons:**
- Higher memory overhead
- Slightly more complex error handling

## Choosing an Executor

With `parallelization=None` (the default), PyFlowReg intersects the executors supported by the configured flow backend with those available on the system and selects the first of multiprocessing, threading, sequential that remains:

```python
# Auto-selection preference: multiprocessing -> threading -> sequential
compensate_recording(options)  # Uses multiprocessing for the flowreg backend
```

Manual selection overrides auto-detection:

```python
config = RegistrationConfig(parallelization="threading")  # Force threading
```

## Flow backends and executors

Each flow backend declares which executors it supports, and PyFlowReg restricts execution to that set. This table is the single reference for backend/executor compatibility; other pages link here rather than restating it.

| Flow backend | Supported executors | Notes |
| --- | --- | --- |
| `flowreg` (default) | sequential, threading, multiprocessing | Built-in NumPy/Numba variational solver |
| `diso` | sequential, threading | OpenCV DIS wrapper; requires `cv2`; `gc` data term only, no GNC |
| `flowreg_torch` | sequential | PyTorch level-solver variant; requires `torch` |
| `flowreg_cuda` | sequential | CuPy/CUDA level-solver variant; requires `cupy` |

The `diso`, `flowreg_torch`, and `flowreg_cuda` backends are registered only when their optional package (`cv2`, `torch`, `cupy` respectively) is importable.

### flowreg (default)

The built-in NumPy/Numba backend supports all three executors:

```python
options = OFOptions(flow_backend="flowreg")

# All of these work:
config = RegistrationConfig(parallelization="sequential")
config = RegistrationConfig(parallelization="threading")
config = RegistrationConfig(parallelization="multiprocessing")
```

### diso

The OpenCV DIS backend supports the sequential and threading executors only:

```python
options = OFOptions(flow_backend="diso")
config = RegistrationConfig(parallelization="threading")
```

### GPU backends (`flowreg_torch`, `flowreg_cuda`)

The GPU backends support the sequential executor only:

```python
from pyflowreg.motion_correction import OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

options = OFOptions(
    flow_backend="flowreg_torch",
    backend_params={"device": "cuda"}
)

config = RegistrationConfig(parallelization="sequential")
```

**Automatic fallback:** If you request an executor a backend does not support, PyFlowReg emits a warning and falls back, preferring multiprocessing, then threading, then sequential among the usable executors:

```python
options = OFOptions(flow_backend="flowreg_cuda")

# Requesting multiprocessing with a backend that supports sequential only
config = RegistrationConfig(parallelization="multiprocessing")

# Warning: Backend 'flowreg_cuda' does not support 'multiprocessing' executor.
# Supported executors: ['sequential']. Falling back to 'sequential'.
compensate_recording(options, config=config)  # Uses sequential
```

## Configuration

### RegistrationConfig Parameters

```python
@dataclass
class RegistrationConfig:
    n_jobs: int = -1  # -1 = all cores, N = use N workers
    verbose: bool = False  # Verbose logging
    parallelization: Optional[str] = None  # None=auto, 'sequential', 'threading', or 'multiprocessing'
```

### Number of Workers

```python
config = RegistrationConfig(
    n_jobs=-1  # Use all CPU cores
)

# Or specify exact number:
config = RegistrationConfig(
    n_jobs=4  # Use 4 workers
)
```

**Guidelines:**
- `n_jobs=-1`: Use all available CPUs (recommended)
- `n_jobs=N`: Use N worker processes/threads
- Sequential ignores `n_jobs` (always 1)

### Buffer Size

The buffer size controls how many frames are read and processed per batch. This is configured in `OFOptions`, not `RegistrationConfig`:

```{literalinclude} ../snippets/user_guide/parallelization/buffer_size.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

**Buffer size tradeoffs:**
- **Larger buffers** (500-1000):
  - Fewer I/O operations
  - Higher memory usage
  - Better for fast storage (SSD)
- **Smaller buffers** (50-100):
  - More frequent I/O
  - Lower memory usage
  - Better for limited RAM or slow storage (HDD)

### Complete Configuration

```python
from pyflowreg.motion_correction import compensate_recording, OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

options = OFOptions(
    input_file="large_video.h5",
    output_path="results/",
    quality_setting="balanced",
    buffer_size=400  # Batch size from MATLAB
)

config = RegistrationConfig(
    parallelization="multiprocessing",  # Executor selection
    n_jobs=-1  # All CPU cores
)

compensate_recording(options, config=config)
```

## Default Behavior

If no config is provided, defaults are used:

```python
# These are equivalent:
compensate_recording(options)
compensate_recording(options, config=RegistrationConfig())
```

Default values:
- `n_jobs=-1` (all cores)
- `verbose=False`
- `parallelization=None` (auto-select)

## Performance Tips

### Memory Management

**For limited RAM systems:**
```python
# Reduce memory usage
config = RegistrationConfig(
    parallelization="threading",  # Lower memory than multiprocessing
    n_jobs=4  # Limit workers
)

options = OFOptions(
    buffer_size=50  # Smaller batches
)
```

**For high-RAM systems:**
```python
# Maximize throughput
config = RegistrationConfig(
    parallelization="multiprocessing",
    n_jobs=-1  # All cores
)

options = OFOptions(
    buffer_size=500  # Larger batches
)
```

### CPU Utilization

**Check CPU usage during processing:**
- Sequential: 100% on single core
- Threading: Distributed but limited by GIL
- Multiprocessing: Near 100% on all cores (best)

**Reserve cores for other tasks:**
```python
import os
n_cores = os.cpu_count()
config = RegistrationConfig(n_jobs=n_cores - 2)  # Leave 2 cores free
```

### I/O Optimization

**For SSD/fast storage:**
```python
# Multiprocessing with large buffers
config = RegistrationConfig(parallelization="multiprocessing")
options = OFOptions(buffer_size=500)
```

**For HDD/slow storage:**
```python
# Threading to avoid I/O contention
config = RegistrationConfig(parallelization="threading")
options = OFOptions(buffer_size=100)
```

## Examples

### Fast Preview Processing

```python
# Quick preview with minimal resources
options = OFOptions(
    input_file="video.h5",
    quality_setting="fast",
    buffer_size=50
)

config = RegistrationConfig(parallelization="sequential")
compensate_recording(options, config=config)
```

### Production Processing

```python
# Maximum performance for production
options = OFOptions(
    input_file="large_dataset.h5",
    quality_setting="balanced",
    buffer_size=500
)

config = RegistrationConfig(
    parallelization="multiprocessing",
    n_jobs=-1
)

compensate_recording(options, config=config)
```

### Memory-Constrained System

```python
# Optimize for limited RAM
options = OFOptions(
    input_file="video.h5",
    quality_setting="balanced",
    buffer_size=100
)

config = RegistrationConfig(
    parallelization="threading",
    n_jobs=4
)

compensate_recording(options, config=config)
```

## Executor Registration

PyFlowReg uses a runtime registry for parallelization executors. Executors auto-register on import:

```{literalinclude} ../snippets/user_guide/parallelization/executor_registry.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## Troubleshooting

### High Memory Usage

**Problem:** Memory usage exceeds available RAM

**Solutions:**
1. Reduce `buffer_size` in OFOptions
2. Switch to threading or sequential
3. Reduce `n_jobs`

```python
# Low-memory configuration
config = RegistrationConfig(
    parallelization="sequential",
    n_jobs=1
)
options = OFOptions(buffer_size=50)
```

### Slow Processing

**Problem:** Processing slower than expected

**Solutions:**
1. Use multiprocessing instead of threading
2. Increase `buffer_size` if RAM allows
3. Increase `n_jobs`
4. Use `quality_setting="fast"` for preview

```python
# Fast processing configuration
config = RegistrationConfig(
    parallelization="multiprocessing",
    n_jobs=-1
)
options = OFOptions(
    quality_setting="fast",
    buffer_size=500
)
```

### Process Hangs or Crashes

**Problem:** Processing hangs or workers crash

**Solutions:**
1. Switch to sequential for debugging
2. Reduce `n_jobs` to avoid resource contention
3. Check for memory issues
4. Verify input file integrity

```python
# Debug configuration
config = RegistrationConfig(
    parallelization="sequential"  # Easier to debug
)
```
