# Quickstart Guide

This guide will get you started with PyFlowReg using both array-based and file-based motion correction workflows.

## Basic Array-Based Workflow

The simplest way to use PyFlowReg is with in-memory arrays using `compensate_arr`:

```{literalinclude} snippets/quickstart/array_workflow.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

Here `alpha=4` increases the smoothness regularization over the default `(1.5, 1.5)` (a scalar is expanded to a 2-tuple). `compensate_arr` always returns the registered frames and displacement fields as in-memory arrays; see the [API reference](api/motion_correction.md) for details.

### Quality Settings

PyFlowReg provides preset quality configurations that control the finest pyramid level the solver computes. Level 0 (L0) is the finest, full-resolution level; higher levels are coarser:

- `quality_setting="fast"` - Solves down to pyramid level 6, suitable for previews
- `quality_setting="balanced"` - Solves down to pyramid level 4
- `quality_setting="quality"` - Solves down to pyramid level 0 (full resolution); the default

Finer pyramid levels capture smaller motion details but require more computation time. Explicitly setting `min_level` overrides the preset and switches `quality_setting` to `"custom"`.

## File-Based Workflow

For large datasets, use `compensate_recording` with file-based I/O:

```{literalinclude} snippets/quickstart/file_workflow.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## Parallel Processing

Batch processing runs through a parallelization executor: `sequential`, `threading`, or `multiprocessing`. By default, PyFlowReg auto-selects the best executor that is supported by the configured flow backend and available on the system, preferring multiprocessing, then threading, then sequential. To manually configure:

```{literalinclude} snippets/quickstart/parallel_processing.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

See the [Parallelization Guide](user_guide/parallelization.md) for executor details and constraints.

**GPU Acceleration:** GPU flow backends are available as `flowreg_torch` (PyTorch) and `flowreg_cuda` (CuPy). Install the dependencies with `pip install pyflowreg[gpu]` and set `flow_backend="flowreg_torch"` or `flow_backend="flowreg_cuda"` in `OFOptions`. See the [Flow Backends Guide](user_guide/backends.md) and the [Parallelization Guide](user_guide/parallelization.md) for backend and executor constraints.

## Multi-Session Processing

For experiments with multiple recordings from the same field of view, use the session processing pipeline:

```{literalinclude} snippets/quickstart/multi_session.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

Or use the command-line interface:
```bash
# Run complete pipeline
pyflowreg-session run --config session.toml

# Or run stages individually (useful for HPC)
pyflowreg-session run --config session.toml --stage 1
pyflowreg-session run --config session.toml --stage 2
pyflowreg-session run --config session.toml --stage 3
```

See the [Multi-Session Processing Guide](user_guide/multi_session.md) for details on HPC integration and advanced configuration.

## Examples and Notebooks

The repository contains demo scripts under `examples/` and demo notebooks under `notebooks/`; the demos with the Jupiter sequence should run out of the box. See the [Examples Gallery](user_guide/examples_gallery.md) for a description of each script and notebook.

Run example scripts as modules from the project root, for example:

```bash
python -m examples.jupiter_demo_arr
```

## Configuration Options

### Key Parameters

```{literalinclude} snippets/quickstart/key_parameters.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## Supported File Formats

PyFlowReg supports multiple file formats through its modular I/O system:

- **HDF5** (.h5, .hdf5) - Recommended for large datasets
- **TIFF** (.tif, .tiff) - Standard microscopy format
- **MAT** (.mat) - MATLAB compatibility
- **MDF** (.mdf) - Sutter MScan format, read via the `MCSX.Data` COM interface (Windows only)

## Next Steps

- See the [User Guide](user_guide/index.md) for detailed workflows
- Check the [API Reference](api/index.md) for complete function documentation
- Read the [Theory](theory/index.md) section to understand the optical flow algorithm
