# API Reference

This section contains the complete API reference for PyFlowReg, automatically generated from docstrings.

## High-Level APIs

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} Motion Correction
:link: motion_correction
:link-type: doc

High-level functions for applying motion correction to videos and arrays.
- `compensate_arr()` - Array-based workflow
- `compensate_recording()` - File-based workflow
- `FlowRegLive` - Real-time processing
- `OFOptions` - Configuration system
:::

:::{grid-item-card} Session Processing
:link: session
:link-type: doc

Multi-recording session processing with inter-sequence alignment.
- `SessionConfig` - Session configuration
- `run_stage1()` - Per-recording correction
- `run_stage2()` - Inter-sequence alignment
- `run_stage3()` - Valid mask computation
:::

:::{grid-item-card} Core Algorithms
:link: core
:link-type: doc

Low-level optical flow computation engine.
- `get_displacement()` - Main optical flow API
- `compute_flow()` - Pyramid level solver
- Warping utilities
- Motion tensor computation
:::

::::

## Utilities

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} I/O System
:link: io
:link-type: doc

File format readers and writers.
- HDF5, TIFF, MAT, MDF support
- Multi-file handling
- VideoReader/VideoWriter interface
:::

:::{grid-item-card} Image Processing
:link: utilities
:link-type: doc

Image processing utilities.
- Pyramid resizing
- Gaussian filtering
- Cross-correlation pre-alignment
- Visualization tools
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

motion_correction
session
core
io
utilities
```
