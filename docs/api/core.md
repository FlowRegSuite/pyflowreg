# Core Algorithms

Low-level optical flow computation engine implementing variational optical flow with a multi-scale pyramid approach.

## When to use the core API

For standard motion-correction workflows, use the high-level APIs from
{doc}`motion_correction`: {func}`~pyflowreg.motion_correction.compensate_arr`
(in-memory arrays) and {func}`~pyflowreg.motion_correction.compensate_recording`
(file-based). They handle video I/O, preprocessing, batch processing, and
executor-based parallelization on top of the core engine.

{func}`~pyflowreg.core.optical_flow.get_displacement` is the per-frame-pair
engine underneath: it computes a dense displacement field of shape `(H, W, 2)`
between a fixed (reference) image and a moving image, where `w[..., 0]` is the
horizontal (x) and `w[..., 1]` the vertical (y) displacement. Use it directly
when building custom pipelines, for example streaming registration in the
style of {class}`~pyflowreg.motion_correction.FlowRegLive` (which obtains its
displacement function via `OFOptions.resolve_get_displacement()` and warps
frames with {func}`~pyflowreg.core.warping.imregister_wrapper`), or when
experimenting with solver parameters.

The core package also provides a [backend registry](#backend-registry) that
maps flow backend names to factories returning
`get_displacement`-compatible callables. Built-in backends registered on
import of `pyflowreg.core`:

| Flow backend | Implementation | Registered when | Supported executors |
|---|---|---|---|
| `flowreg` | Variational solver ({func}`~pyflowreg.core.optical_flow.get_displacement`) | always | all |
| `diso` | OpenCV DIS optical flow ({class}`~pyflowreg.core.diso_optical_flow.DisoOF`) | `cv2` importable | `sequential`, `threading` |
| `flowreg_torch` | Variational solver with PyTorch level solver | `torch` importable | `sequential` |
| `flowreg_cuda` | Variational solver with CuPy level solver | `cupy` importable | `sequential` |

## Optical Flow

```{eval-rst}
.. automodule:: pyflowreg.core.optical_flow
   :no-members:
```

### Main API

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.get_displacement
```

### Graduated Non-Convexity Helpers

Graduated non-convexity (GNC) is opt-in via the `gnc_schedule` parameter of
{func}`~pyflowreg.core.optical_flow.get_displacement`. A schedule is a 1D
sequence of stage weights in `[0, 1]` that must be monotone nondecreasing,
contain at least two stages, start at `0.0` (quadratic penalty), and end at
`1.0` (fully robust penalty). Each stage reruns the pyramid with the previous
stage result as initialization; the number of warp/relinearize steps per
pyramid level defaults to 10 in GNC mode and can be overridden with
`warping_steps`.

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.normalize_gnc_schedule
```

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.normalize_warping_steps
```

### Motion Tensor Computation

The motion tensor function is selected through the `const_assumption`
parameter of {func}`~pyflowreg.core.optical_flow.get_displacement`:
`'gc'`/`'gradient'` (default, gradient constancy),
`'gray'`/`'brightness'` (gray-value constancy), and `'cs'`/`'census'`
(census constancy).

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.get_motion_tensor_gc
```

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.get_motion_tensor_gray
```

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.get_motion_tensor_cs
```

### Boundary Handling

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.add_boundary
```

### Level Solver Dispatch

`level_solver` is the default `level_solver_backend` used by
{func}`~pyflowreg.core.optical_flow.get_displacement`. It dispatches to
{func}`~pyflowreg.core.level_solver.compute_flow`, or to
{func}`~pyflowreg.core.level_solver.compute_flow_gnc` when a GNC stage weight
(`gnc_beta`) is given, and returns the flow increments `(du, dv)`. The
`flowreg_torch` and `flowreg_cuda` backends replace this function with their
own level solvers.

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.level_solver
```

## Warping and Registration

```{eval-rst}
.. automodule:: pyflowreg.core.warping
   :no-members:
```

```{eval-rst}
.. autofunction:: pyflowreg.core.warping.imregister_wrapper
```

```{eval-rst}
.. autofunction:: pyflowreg.core.warping.warpingDepth
```

```{eval-rst}
.. autofunction:: pyflowreg.core.warping.align_sequence
```

```{eval-rst}
.. autofunction:: pyflowreg.core.warping.backward_valid_mask
```

```{eval-rst}
.. autofunction:: pyflowreg.core.warping.imregister_binary
```

```{eval-rst}
.. autofunction:: pyflowreg.core.warping.compute_batch_valid_masks
```

## Pyramid Level Solver

```{eval-rst}
.. automodule:: pyflowreg.core.level_solver
   :no-members:
```

### Flow Computation

```{eval-rst}
.. autofunction:: pyflowreg.core.level_solver.compute_flow
```

```{eval-rst}
.. autofunction:: pyflowreg.core.level_solver.compute_flow_gnc
```

### Boundary Conditions

```{eval-rst}
.. autofunction:: pyflowreg.core.level_solver.set_boundary_2d
```

### Nonlinearity Functions

```{eval-rst}
.. autofunction:: pyflowreg.core.level_solver.nonlinearity_smoothness_2d
```

## Backend Registry

The registry maps flow backend names to factory functions. A factory
returns a callable with the same interface as
{func}`~pyflowreg.core.optical_flow.get_displacement`; when a backend is
resolved through `OFOptions`, the entries of `OFOptions.backend_params` are
forwarded to the factory as keyword arguments. Each backend additionally
declares which parallelization executors it supports, using the exact
executor names `'sequential'`, `'threading'`, `'multiprocessing'`,
`'multiprocessing_fork'`, and `'multiprocessing_spawn'`; if no set is given
at registration, all executors are assumed to be supported.

{func}`~pyflowreg.core.backend_registry.register_backend`,
{func}`~pyflowreg.core.backend_registry.get_backend`,
{func}`~pyflowreg.core.backend_registry.list_backends`, and
{func}`~pyflowreg.core.backend_registry.is_backend_available` are re-exported
from `pyflowreg.core`;
{func}`~pyflowreg.core.backend_registry.get_backend_executors` is available
from `pyflowreg.core.backend_registry`.

```{eval-rst}
.. automodule:: pyflowreg.core.backend_registry
   :no-members:
```

```{eval-rst}
.. autofunction:: pyflowreg.core.backend_registry.register_backend
```

```{eval-rst}
.. autofunction:: pyflowreg.core.backend_registry.get_backend
```

```{eval-rst}
.. autofunction:: pyflowreg.core.backend_registry.list_backends
```

```{eval-rst}
.. autofunction:: pyflowreg.core.backend_registry.is_backend_available
```

```{eval-rst}
.. autofunction:: pyflowreg.core.backend_registry.get_backend_executors
```

## DISO Backend

The `diso` flow backend wraps OpenCV's Dense Inverse Search (DIS) optical
flow (`cv2.DISOpticalFlow`) behind the same callable interface as
{func}`~pyflowreg.core.optical_flow.get_displacement`. It is registered only
when OpenCV (`cv2`) is importable. {class}`~pyflowreg.core.diso_optical_flow.DisoOF`
reduces multi-channel input to grayscale using the channel weights, accepts
an optional initial flow field for warm starts, and initializes the OpenCV
DIS object lazily so instances remain picklable.

Compared to the variational `flowreg` backend, `diso` has the following
restrictions (enforced when the backend is resolved through `OFOptions`):

- Only the default `'gc'` constancy setting is accepted; requesting `'gray'`
  or `'cs'` raises `ValueError`.
- Graduated non-convexity is not supported; setting `gnc_schedule` or
  `warping_steps` raises `ValueError`.
- Supported executors are `sequential` and `threading`. Multiprocessing
  workers import the variational solver directly and do not reconstruct
  registry backends, so `diso` is restricted to in-process executors.

```{eval-rst}
.. automodule:: pyflowreg.core.diso_optical_flow
   :members:
```
