# Core Algorithms

Low-level optical flow computation engine implementing variational optical flow with multi-scale pyramid approach.

## Optical Flow

```{eval-rst}
.. automodule:: pyflowreg.core.optical_flow
```

### Main API

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.get_displacement
```

### Warping and Registration

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.imregister_wrapper
```

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.warpingDepth
```

### Motion Tensor Computation

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.get_motion_tensor_gc
```

### Boundary Handling

```{eval-rst}
.. autofunction:: pyflowreg.core.optical_flow.add_boundary
```

## Pyramid Level Solver

```{eval-rst}
.. automodule:: pyflowreg.core.level_solver
```

### Flow Computation

```{eval-rst}
.. autofunction:: pyflowreg.core.level_solver.compute_flow
```

### Boundary Conditions

```{eval-rst}
.. autofunction:: pyflowreg.core.level_solver.set_boundary_2d
```

### Nonlinearity Functions

```{eval-rst}
.. autofunction:: pyflowreg.core.level_solver.nonlinearity_smoothness_2d
```

## DISO Backend

```{eval-rst}
.. automodule:: pyflowreg.core.diso_optical_flow
   :members:
```
