# Utilities

Image processing and helper utilities for motion correction workflows.

## Image Processing

Normalization and Gaussian filtering helpers used by the motion correction pipeline, including a 1D half-kernel filter for causal temporal filtering in online processing.

```{eval-rst}
.. automodule:: pyflowreg.util.image_processing
   :members:
```

## Pyramid Resizing

Image resizing utilities for pyramid construction, including a fused Gaussian-cubic interpolation that applies Gaussian smoothing during downsampling to reduce aliasing.

```{eval-rst}
.. automodule:: pyflowreg.util.resize_util
   :members:
```

## Cross-Correlation Pre-Alignment

Rigid shift estimation via phase cross-correlation, used to optionally pre-align frames before the variational optical flow refinement. See [Pre-Alignment](../user_guide/prealignment.md) for usage.

```{eval-rst}
.. automodule:: pyflowreg.util.xcorr_prealignment
   :members:
```

## Visualization

Displacement field visualization: color-coded flow maps and quiver overlays. `quiver_visualization()` requires the optional `matplotlib` dependency.

```{eval-rst}
.. automodule:: pyflowreg.util.visualization
   :members:
```

## Download Helpers

Download utilities for the demo datasets used in the examples.

```{eval-rst}
.. automodule:: pyflowreg.util.download
   :members:
```

## Superresolution Helpers

Helpers for warping frames on an upsampled grid with a correspondingly scaled displacement field.

```{eval-rst}
.. automodule:: pyflowreg.util.superresolution_helpers
   :members:
```
