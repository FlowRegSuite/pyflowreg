"""
Utility Functions for Image Processing, I/O, and Visualization
===============================================================

This module provides supporting utilities for microscopy data processing,
including image manipulation, file I/O, visualization, and preprocessing.

Submodules
----------
io
    Modular I/O system supporting multiple formats (HDF5, TIFF, MAT, MDF)
    through VideoReader/VideoWriter interfaces
resize_util
    Efficient aliasing-free pyramid resizing functions. The optical flow engine
    uses imresize_fused_gauss_cubic, which fuses Gaussian anti-aliasing and
    bicubic interpolation in a single numba-optimized pass for maximum performance.
    Also provides alternative resize methods including pure bicubic (imresize_numba),
    OpenCV-based variants, and scikit-image integration.
image_processing
    Image manipulation, filtering, normalization, and warping functions
visualization
    Motion field visualization and quality control plotting
xcorr_prealignment
    Cross-correlation based reference frame alignment
superresolution_helpers
    Helper functions for super-resolution microscopy applications
download
    Dataset download utilities

See Also
--------
pyflowreg.core : Optical flow computation engine
pyflowreg.motion_correction : High-level motion correction APIs
"""

__all__ = []
