# Flow Backends

A **flow backend** is the implementation that computes the displacement field
between a single frame pair (reference and moving frame). It is independent of
the **executor**, which controls how that computation is parallelized across
the frames of a recording (sequential, threading, multiprocessing). Backends
declare which executors they support, so the two choices interact; see
[Parallelization](parallelization.md) for executor selection and the
backend/executor compatibility rules.

## Available Backends

Backends register themselves when `pyflowreg.core` is imported. Backends with
optional dependencies are only registered when their dependency is importable.

| Backend | What it is | Available when | Supported executors | Constraints |
|---|---|---|---|---|
| `flowreg` | Default variational optical flow solver (NumPy/Numba) | always | `sequential`, `threading`, `multiprocessing` | none |
| `diso` | OpenCV Dense Inverse Search (DIS) optical flow wrapper | `cv2` importable | `sequential`, `threading` | `'gc'` constancy only; no `gnc_schedule`/`warping_steps` |
| `flowreg_torch` | Variational solver with a PyTorch level solver (CPU or GPU) | `torch` importable | `sequential` | — |
| `flowreg_cuda` | Variational solver with a CuPy/CUDA level solver | `cupy` importable | `sequential` | — |

Notes:

- `diso` only passes validation with the default gradient-constancy data term
  (`constancy_assumption='gc'`). Requesting `'gray'` or `'cs'`, or setting
  `gnc_schedule` or `warping_steps`, raises `ValueError` when the backend is
  resolved from `OFOptions`.
- `diso` excludes multiprocessing because the multiprocessing workers import
  the variational solver directly and do not reconstruct registry backends.
- `diso` honors the flow initialization (`uv`) the pipelines pass between
  batches and frames, forwarding it to OpenCV DIS as the initial flow; the
  other variational solver keywords (`alpha`, `iterations`, ...) are accepted
  but ignored.

You can inspect what is registered in the current environment:

```{literalinclude} ../snippets/user_guide/backends/inspect_backends.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## Selecting a Backend

The backend is selected through `OFOptions`; backend-specific settings go into
the `backend_params` dictionary, which is forwarded to the backend factory as
keyword arguments:

```python
from pyflowreg.motion_correction import OFOptions, compensate_recording

options = OFOptions(
    input_file="video.h5",
    output_path="results/",
    flow_backend="flowreg_torch",
    backend_params={"device": "cuda", "dtype": "float32"},
)
compensate_recording(options)
```

If `flow_backend` names a backend that is not registered (for example because
its optional dependency is missing), resolving the backend raises a
`ValueError` that lists the registered backends.

### Executor interplay

When `RegistrationConfig(parallelization=None)` (the default) is used, the
executor is auto-selected from the intersection of the executors supported by
the chosen backend and the executors available on the system, preferring
multiprocessing, then threading, then sequential. Explicitly requesting an
executor the backend does not support triggers a warning and a fallback to a
compatible executor. See [Parallelization](parallelization.md) for details,
including the GPU-backend constraint.

## backend_params Reference

`flowreg` ignores `backend_params`; the solver parameters of the variational
method (`alpha`, `iterations`, `levels`, ...) are regular `OFOptions` fields,
not backend parameters.

### `diso`

Keys map one-to-one onto the `DisoOF` constructor:

| Key | Default | Meaning |
|---|---|---|
| `preset` | `cv2.DISOPTICAL_FLOW_PRESET_MEDIUM` | OpenCV DIS preset (`ULTRAFAST`, `FAST`, or `MEDIUM`) |
| `finest_scale` | `2` | Finest pyramid scale the flow is computed on (0 = full resolution) |
| `gradient_descent_iterations` | `12` | Gradient descent iterations in the patch inverse search |
| `patch_size` | `8` | Patch size in pixels |
| `patch_stride` | `4` | Stride between neighboring patches |
| `use_mean_normalization` | `True` | Mean-normalized patch distances |
| `use_spatial_propagation` | `True` | Spatial propagation of flow vectors |

Unknown keys raise a `TypeError`.

```python
import cv2

options = OFOptions(
    flow_backend="diso",
    backend_params={"preset": cv2.DISOPTICAL_FLOW_PRESET_FAST, "finest_scale": 1},
)
```

### `flowreg_torch`

| Key | Default | Meaning |
|---|---|---|
| `device` | `None` | PyTorch device string (e.g. `"cuda"`, `"cuda:0"`, `"cpu"`). `None` selects `"cuda"` if available, otherwise `"cpu"`. If a CUDA device is requested but unavailable, a warning is emitted and the backend falls back to CPU. |
| `dtype` | `"float64"` | Tensor dtype, `"float32"` or `"float64"` |

Additional keys are accepted for compatibility and ignored.

### `flowreg_cuda`

The CuPy factory takes no parameters; any `backend_params` entries are
accepted for compatibility and ignored.

## Registering a Custom Backend

`pyflowreg.core.register_backend` maps a backend name to a factory. The
factory is called with the entries of `OFOptions.backend_params` as keyword
arguments and must return a callable with the same interface as
`get_displacement`: it is called with the fixed (reference) image and the
moving image and returns a displacement field `w` of shape `(H, W, 2)`, where
`w[..., 0]` is the horizontal component `u` (x) and `w[..., 1]` the vertical
component `v` (y). The pipelines also pass solver keyword arguments (such as
`alpha`, `iterations`, `uv`, `weight`); a custom backend must accept them, but
may ignore them.

The following example is illustrative:

```python
import numpy as np
from pyflowreg.core import register_backend


def my_factory(**backend_params):
    def my_get_displacement(fixed, moving, **kwargs):
        h, w = fixed.shape[:2]
        flow = np.zeros((h, w, 2), dtype=np.float32)
        # ... compute flow[..., 0] (u, horizontal) and flow[..., 1] (v, vertical)
        return flow

    return my_get_displacement


register_backend(
    "my_backend",
    my_factory,
    supported_executors={"sequential", "threading"},
)

options = OFOptions(flow_backend="my_backend", backend_params={})
```

If `supported_executors` is omitted, the backend is assumed to support all
executors. Multiprocessing workers import the built-in variational solver
directly and do not reconstruct registry backends, so custom backends should
not declare multiprocessing support. The registry is in-memory: the module
that calls `register_backend` must be imported in the current process before
the options are resolved.

For one-off experiments there is a shortcut that bypasses the registry:
`OFOptions.get_displacement_factory` (a factory called with
`backend_params`) and `OFOptions.get_displacement_impl` (a ready-made
callable) override `flow_backend` when set. Both are excluded from
serialization.

## See Also

- [Core API reference](../api/core.md) - `get_displacement`, the backend
  registry functions, and the DISO wrapper
- [Parallelization](parallelization.md) - executors and backend/executor
  compatibility
