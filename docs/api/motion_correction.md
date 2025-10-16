# Motion Correction

High-level APIs for applying motion correction and motion analysis to microscopy videos.

## Core Concepts

### Output Formats

PyFlowReg supports flexible output handling through the `output_format` parameter:

- **`OutputFormat.ARRAY`** - Accumulate in memory, return as array (default for array workflow)
- **`OutputFormat.NULL`** - Discard output, callback-only processing (no storage overhead)
- **`OutputFormat.HDF5`** - HDF5 file storage
- **`OutputFormat.TIFF`** - TIFF stack output
- **`OutputFormat.MAT`** - MATLAB compatible files

### Callback System

All motion correction functions support real-time data access through callbacks:

| Callback | Signature | Description |
|----------|-----------|-------------|
| `progress_callback` | `(current: int, total: int) -> None` | Progress updates |
| `w_callback` | `(w_batch: ndarray, start_idx: int, end_idx: int) -> None` | Access displacement fields during processing |
| `registered_callback` | `(batch: ndarray, start_idx: int, end_idx: int) -> None` | Access corrected frames during processing |

Callbacks enable:
- Real-time visualization without waiting for completion
- Motion tracking and analysis during processing
- Memory-efficient processing with `OutputFormat.NULL`
- Integration with visualization tools like napari

## Array-Based Workflow

The primary function for in-memory motion correction with callback support:

```python
compensate_arr(
    c1: np.ndarray,                    # Video to correct
    c_ref: np.ndarray,                  # Reference frame
    options: OFOptions = None,          # Configuration
    progress_callback: Callable = None, # Progress updates
    w_callback: Callable = None,        # Displacement access
    registered_callback: Callable = None # Corrected frame access
) -> Tuple[np.ndarray, np.ndarray]     # Returns (registered, w)
```

### Example with Callbacks

```python
import numpy as np
from pyflowreg.motion_correction import compensate_arr
from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat

def track_motion(w_batch, start_idx, end_idx):
    """Process displacement fields as they're computed."""
    for i in range(w_batch.shape[0]):
        magnitude = np.sqrt(w_batch[i, :, :, 0]**2 + w_batch[i, :, :, 1]**2)
        print(f"Frame {start_idx + i}: mean motion = {np.mean(magnitude):.2f}")

# Configure for callback-only processing
options = OFOptions(
    output_format=OutputFormat.NULL,  # No storage
    save_w=True,                       # Compute displacement fields
    buffer_size=20                     # Process 20 frames at a time
)

# Run with callbacks
registered, w = compensate_arr(
    video, reference, options,
    w_callback=track_motion
)
```

```{eval-rst}
.. autofunction:: pyflowreg.motion_correction.compensate_arr
```

## File-Based Workflow

File-based processing with callback support through `BatchMotionCorrector`:

```python
from pyflowreg.motion_correction.compensate_recording import BatchMotionCorrector
from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat

class ProcessingMonitor:
    def __init__(self):
        self.batch_count = 0

    def on_batch_complete(self, batch, start_idx, end_idx):
        self.batch_count += 1
        print(f"Batch {self.batch_count} complete: frames {start_idx}-{end_idx}")

monitor = ProcessingMonitor()

options = OFOptions(
    input_file="recording.h5",
    output_format=OutputFormat.HDF5,
    output_path="results/",
    save_w=True
)

compensator = BatchMotionCorrector(options)
compensator.register_registered_callback(monitor.on_batch_complete)
compensator.run()
```

```{eval-rst}
.. autofunction:: pyflowreg.motion_correction.compensate_recording
```

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.compensate_recording.RegistrationConfig
   :members:
```

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.compensate_recording.BatchMotionCorrector
   :members:
```

## Real-Time Processing

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.FlowRegLive
   :members:
```

## Configuration

### OutputFormat Enum

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.OF_options.OutputFormat
   :members:
   :undoc-members:
```

Key output formats:
- **`OutputFormat.NULL`** - Discards output, ideal for callback-only processing
- **`OutputFormat.ARRAY`** - Returns in-memory arrays (default for `compensate_arr`)
- **`OutputFormat.HDF5`** - Efficient storage for large datasets
- **`OutputFormat.TIFF`** - Standard microscopy format
- **`OutputFormat.MAT`** - MATLAB compatibility

### OFOptions Class

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.OFOptions
   :members:
   :exclude-members: model_config, model_fields, model_computed_fields
```

## Parallelization

PyFlowReg provides multiple parallelization backends for batch processing.

```{eval-rst}
.. automodule:: pyflowreg.motion_correction.parallelization
   :members:
   :exclude-members: register
```

### Sequential Executor

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.parallelization.SequentialExecutor
   :members:
```

### Threading Executor

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.parallelization.ThreadingExecutor
   :members:
```

### Multiprocessing Executor

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.parallelization.MultiprocessingExecutor
   :members:
```
