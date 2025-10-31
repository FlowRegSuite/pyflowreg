# Changelog

All notable changes to PyFlowReg will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- **Multi-session processing module** with full MATLAB parity
  - Three-stage pipeline for aligning multiple recordings
  - `SessionConfig` with TOML/YAML support
  - Per-recording motion correction with valid mask persistence
  - Inter-sequence alignment via phase cross-correlation and optical flow
  - Session-wide valid mask computation
  - HPC array job support (SLURM, SGE, PBS)
  - Command-line interface: `pyflowreg-session`
  - Crash-safe atomic file writes
  - HDF5 completeness verification
- **Warping utilities** in `pyflowreg.core.warping`
  - `backward_valid_mask()` for in-bounds checking
  - `imregister_binary()` for binary mask warping
  - `compute_batch_valid_masks()` for batch processing
- Comprehensive documentation with Sphinx, user guides, API reference, and theory sections
- Example session configuration files (TOML and YAML)

### Fixed
- Batch size parameter confusion (removed unused parameter from RegistrationConfig)
- Thread oversubscription in multiprocessing executor (now sets thread limits)

## [0.1.0a5] - 2025-10-23

### Added

- **GPU Acceleration**: New `flowreg_cuda` and `flowreg_torch` backends with device-aware resize utilities, runtime detection, `pyflowreg[gpu]` optional extra, and `examples/jupiter_demo_arr_gpu.py` walkthrough (3b962ff, 5010e19, e2c724f, 8b2795a)
- **Napari-oriented APIs**: `BatchMotionCorrector.register_w_callback` / `register_registered_callback` matching `compensate_arr` parameters, plus `NullVideoWriter` and `OutputFormat.NULL` for headless pipelines (d1d39f9, 480d3bc, dbe0d7f)
- **Documentation**: Complete MyST/Sphinx site with installation, quickstart, API reference, theory, workflows, and GPU setup guides, backed by ReadTheDocs config and executable quickstart/parallelization tests (fd5d778, 288752f, 75dd88a)

### Fixed

- **Critical weight handling**: Multi-dimensional weight handling in `OFOptions` now correctly preserves numpy arrays, rejects invalid 4D arrays, and properly handles 2D `(H, W)` and 3D `(H, W, C)` spatial weight maps (f01f880)
- **Executor registration**: Restored side-effect imports for multiprocessing/threading executor registration that were removed by pre-commit hooks, preventing silent fallback to sequential mode (ae16755)
- **Dimensionality checks**: Fixed edge cases in `OF_options` weight and sigma parameter validation (31def15)

### Changed

- **Backend registry**: Now restricts executor choices per backend, improves availability checks, and provides clearer logging so sequential fallback is visible (5010e19, 1d7e8a9, f3785aa)
- **Testing**: Added regression tests for weights, callbacks, executors, and Null writer; GPU-aware test skips for platforms without CuPy (efd072d, fd3b107, b051775)
- **Tooling**: Applied FlowRegSuite pre-commit hooks and reorganized public exports without breaking external API (cedb82e, c2a984e)

## [0.1.0a4]

Fixed batch normalization to use reference values.

## [0.1.0a3]

- Cross-correlation pre-alignment feature
- Backend architecture refactoring
- ScanImage TIFF format compatibility fix

## [0.1.0a2]

- CI/CD improvements and Python 3.13 support
- Demo download utilities
- Refactored CLI and paper benchmarks into separate repositories

## [0.1.0a1]

Initial alpha release with core variational optical flow engine, multi-channel 2D motion correction, and modular I/O system.

[Unreleased]: https://github.com/FlowRegSuite/pyflowreg/compare/v0.1.0a5...HEAD
[0.1.0a5]: https://github.com/FlowRegSuite/pyflowreg/compare/v0.1.0a4...v0.1.0a5
[0.1.0a4]: https://github.com/FlowRegSuite/pyflowreg/compare/v0.1.0a3...v0.1.0a4
[0.1.0a3]: https://github.com/FlowRegSuite/pyflowreg/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/FlowRegSuite/pyflowreg/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/FlowRegSuite/pyflowreg/releases/tag/v0.1.0a1
