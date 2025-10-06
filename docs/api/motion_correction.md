# Motion Correction

High-level APIs for applying motion correction and motion analysis to microscopy videos.

## Array-Based Workflow

```{eval-rst}
.. autofunction:: pyflowreg.motion_correction.compensate_arr
```

## File-Based Workflow

```{eval-rst}
.. autofunction:: pyflowreg.motion_correction.compensate_recording
```

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.compensate_recording.RegistrationConfig
   :members:
```

## Real-Time Processing

```{eval-rst}
.. autoclass:: pyflowreg.motion_correction.FlowRegLive
   :members:
```

## Configuration

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
