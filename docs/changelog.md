# Changelog

All notable changes to PyFlowReg will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation with Sphinx, user guides, API reference, and theory sections

### Fixed
- Batch size parameter confusion (removed unused parameter from RegistrationConfig)

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

[Unreleased]: https://github.com/FlowRegSuite/pyflowreg/compare/v0.1.0a4...HEAD
[0.1.0a4]: https://github.com/FlowRegSuite/pyflowreg/compare/v0.1.0a3...v0.1.0a4
[0.1.0a3]: https://github.com/FlowRegSuite/pyflowreg/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/FlowRegSuite/pyflowreg/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/FlowRegSuite/pyflowreg/releases/tag/v0.1.0a1
