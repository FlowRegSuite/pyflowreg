# User Guide

This guide provides practical how-to documentation for common PyFlowReg workflows and configurations.

## Workflows

Learn how to use PyFlowReg in different processing scenarios:

- **[Array-Based Workflow](workflows.md#array-based-workflow)** - In-memory processing for smaller datasets
- **[File-Based Workflow](workflows.md#file-based-workflow)** - Efficient processing of large video files
- **[Online Processing](online_processing.md)** - Streaming motion correction with adaptive reference (FlowRegLive)
- **[Multi-Session Processing](multi_session.md)** - Aligning multiple recordings across sessions
- **[3D Volumes](3d_volumes.md)** - Volumetric workflows
- **[Z-Alignment](z_align.md)** - Depth-shift estimation and correction for volumetric recordings
- **[Examples Gallery](examples_gallery.md)** - Tour of the demo scripts and notebooks

## Configuration

Detailed guides for configuring motion correction:

- **[OFOptions Reference](configuration.md)** - Complete parameter reference
- **[Quality Settings](configuration.md#quality-settings)** - Speed vs accuracy tradeoffs
- **[Preprocessing Options](configuration.md#preprocessing)** - Binning, filtering, normalization
- **[Reference Selection](configuration.md#reference-selection)** - Choosing reference frames
- **[Flow Backends](backends.md)** - flowreg, diso, and the GPU backends
- **[Cross-Correlation Pre-Alignment](prealignment.md)** - Rigid initialization before variational flow

## File Formats

Working with different file formats:

- **[Supported Formats](file_formats.md)** - HDF5, TIFF, MAT, MDF
- **[Format Conversion](file_formats.md)** - Converting between formats
- **[Multi-File Datasets](file_formats.md)** - Handling multiple files

## Parallelization

Optimizing performance with parallel processing:

- **[Choosing an Executor](parallelization.md)** - Sequential, threading, multiprocessing
- **[Configuration](parallelization.md)** - Tuning worker count and buffer size
- **[Performance Tips](parallelization.md)** - Memory management and optimization

```{toctree}
:maxdepth: 2
:hidden:

workflows
online_processing
multi_session
configuration
backends
prealignment
file_formats
parallelization
3d_volumes
z_align
examples_gallery
```
