# Motion Correction

High-level APIs for applying motion correction and motion analysis to microscopy videos.

## Core Concepts

### Output Formats

For file-based workflows (`compensate_recording`), the output destination is selected through the `output_format` parameter:

- **`OutputFormat.ARRAY`** - Accumulate in memory, return as array
- **`OutputFormat.NULL`** - Discard output, callback-only processing (no storage overhead)
- **`OutputFormat.HDF5`** - HDF5 file storage
- **`OutputFormat.TIFF`** - TIFF stack output
- **`OutputFormat.MAT`** - MATLAB compatible files

### Callback System

Callbacks provide access to results while processing is still running. `compensate_arr` accepts them as keyword arguments; for file-based processing, register them on a `BatchMotionCorrector` instance via `register_progress_callback`, `register_w_callback`, and `register_registered_callback` (the `compensate_recording` convenience wrapper does not take callback arguments).

| Callback | Signature | Description |
|----------|-----------|-------------|
| `progress_callback` | `(current: int, total: int) -> None` | Progress updates |
| `w_callback` | `(w_batch: ndarray, start_idx: int, end_idx: int) -> None` | Access displacement fields during processing |
| `registered_callback` | `(batch: ndarray, start_idx: int, end_idx: int) -> None` | Access corrected frames during processing |

**Execution timing** (see `BatchMotionCorrector.run()`): the pipeline reads the input batch-by-batch. After each batch, `registered_callback` fires once with the compensated frames (after they have been written to the configured output), followed by `w_callback` with the displacement fields of that batch. `progress_callback` fires with cumulative frame counts as frames complete within a batch; with the multiprocessing executor it is updated once per batch rather than per frame. Exceptions raised inside callbacks are caught and reported as warnings, and processing continues.

Callbacks enable:
- Real-time visualization without waiting for completion
- Motion tracking and analysis during processing
- Memory-efficient file-based processing with `OutputFormat.NULL`
- Integration with visualization tools like napari

## Array-Based Workflow

The primary function for in-memory motion correction with callback support (abridged signature; see the full reference below):

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

```{literalinclude} ../snippets/api/motion_correction/callback_example.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

`compensate_arr` always returns arrays in memory and ignores `options.output_format`:
it works on a copy of the options and forces `output_format=OutputFormat.ARRAY`, so any
user-set value has no effect. For callback-only processing without storing results, use
`compensate_recording` with `OFOptions(output_format=OutputFormat.NULL)` instead.

```{eval-rst}
.. autofunction:: pyflowreg.motion_correction.compensate_arr
```

## File-Based Workflow

File-based processing with callback support through `BatchMotionCorrector`:

```{literalinclude} ../snippets/api/motion_correction/file_workflow_callbacks.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
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

The individual formats are described under [Output Formats](#output-formats) above.

### OFOptions Class

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.OFOptions
   :members:
   :exclude-members: model_config, model_fields, model_computed_fields
```

## Parallelization

PyFlowReg provides multiple executors (sequential, threading, multiprocessing) for batch processing. See [Parallelization](../user_guide/parallelization.md) for usage guidance, including executor constraints with GPU flow backends.

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
