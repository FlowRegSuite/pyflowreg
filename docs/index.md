# PyFlowReg

**Python implementation and extension of the Flow-Registration approach for motion correction of 2-photon microscopy videos**

PyFlowReg is in alpha; if you find errors or missing features, please [report an issue](https://github.com/FlowRegSuite/pyflowreg/issues).

PyFlowReg provides high-accuracy motion correction for 2-photon microscopy videos and volumetric 3D scans using variational optical flow techniques. Dense motion information is explicitly computed, enabling both motion-corrected output and subsequent motion analysis or visualization.

This is a Python port of the [Flow-Registration MATLAB toolbox](https://github.com/FlowRegSuite/flow_registration), modeled on the MATLAB implementation while adding Python-specific features and optimizations.

## Key Features

- **Variational model** optimized for motion statistics in 2P microscopy
- **Dense motion fields** returned for analysis and visualization
- **Multi-channel support** with automatic weight normalization
- **Multi-session processing** for longitudinal studies with inter-sequence alignment
- **Parallel processing** with threading and multiprocessing executors
- **HPC integration** with array job support (SLURM, SGE, PBS) in the session pipeline
- **Flexible I/O** supporting HDF5, TIFF, MAT, and MDF formats
- **Rigid pre-alignment** of each frame via cross-correlation, for robustness to large drift
- **MATLAB compatibility** modeled on the MATLAB Flow-Registration implementation

## Getting Started

```{literalinclude} snippets/index/getting_started.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

The default regularization strength is `alpha=(1.5, 1.5)`; pass `alpha` to `OFOptions` to override it.

## Citation

If you use PyFlowReg in your research, please cite:

> P. Flotho, S. Nomura, M. Flotho, A. Keller and B. Kuhn, "Pyflowreg: A python package for high accuracy motion correction of 2P microscopy videos and 3D scans," (in preparation), 2025.

and for Flow-Registration:

> P. Flotho, S. Nomura, B. Kuhn and D. J. Strauss, "Software for Non-Parametric Image Registration of 2-Photon Imaging Data," J Biophotonics, 2022. [doi:10.1002/jbio.202100330](https://doi.org/10.1002/jbio.202100330)

## Related Projects

Part of the [FlowRegSuite](https://github.com/FlowRegSuite) ecosystem:
- [Flow-Registration (MATLAB)](https://github.com/FlowRegSuite/flow_registration) - Original MATLAB implementation
- [Flow-Registration ImageJ/Fiji plugin](https://github.com/FlowRegSuite/flow_registration_IJ)
- [napari-flowreg](https://github.com/FlowRegSuite/napari-flowreg) - Interactive visualization plugin
- [pyflowreg-mcp](https://github.com/FlowRegSuite/pyflowreg-mcp) - MCP tools for LLM workflows

## Documentation

```{toctree}
:maxdepth: 2

installation
quickstart
api/index
user_guide/index
theory/index
troubleshooting
changelog
```

## License

PyFlowReg is released under the CC BY-NC-SA 4.0 license.
