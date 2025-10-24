# User Guide

This guide provides practical how-to documentation for common PyFlowReg workflows and configurations.

## Workflows

Learn how to use PyFlowReg in different processing scenarios:

- **[Array-Based Workflow](workflows.md#array-based-workflow)** - In-memory processing for smaller datasets
- **[File-Based Workflow](workflows.md#file-based-workflow)** - Efficient processing of large video files
- **[Real-Time Processing](workflows.md#real-time-processing)** - Online motion correction with adaptive reference
- **[Multi-Session Processing](multi_session.md)** - Aligning multiple recordings across sessions

## Configuration

Detailed guides for configuring motion correction:

- **[OFOptions Reference](configuration.md)** - Complete parameter reference
- **[Quality Settings](configuration.md#quality-settings)** - Speed vs accuracy tradeoffs
- **[Preprocessing Options](configuration.md#preprocessing)** - Binning, filtering, normalization
- **[Reference Selection](configuration.md#reference-selection)** - Choosing reference frames

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
multi_session
configuration
file_formats
parallelization
3d_volumes
```
