# PyFlowReg

**Variational Optical Flow for 2-Photon Microscopy**

PyFlowReg provides high-accuracy motion correction for 2-photon microscopy videos and volumetric 3D scans using variational optical flow techniques. Dense motion information is explicitly computed, enabling both motion-corrected output and subsequent motion analysis or visualization.

This is a Python port of the [Flow-Registration MATLAB toolbox](https://github.com/FlowRegSuite/flow_registration), maintaining full algorithmic compatibility while adding modern Python features and optimizations.

## Key Features

- **Variational model** optimized for motion statistics in 2P microscopy
- **Dense motion fields** returned for analysis and visualization
- **Multi-channel support** with automatic weight normalization
- **Parallel processing** with threading and multiprocessing executors
- **Flexible I/O** supporting HDF5, TIFF, MAT, and MDF formats
- **Reference pre-alignment** via cross-correlation for improved robustness
- **MATLAB compatibility** maintaining algorithmic parity with Flow-Registration

## Getting Started

```python
import numpy as np
from pyflowreg.motion_correction import compensate_arr, OFOptions

# Load your 2-photon video (T, H, W, C)
video = np.load("my_video.npy")
reference = np.mean(video[:10], axis=0)

# Configure and run motion correction
options = OFOptions(quality_setting="balanced")
registered, flow = compensate_arr(video, reference, options)
```

## Citation

If you use PyFlowReg in your research, please cite:

> "Pyflowreg," (in preparation), 2025.

and for Flow-Registration:

> P. Flotho, S. Nomura, B. Kuhn and D. J. Strauss, "Software for Non-Parametric Image Registration of 2-Photon Imaging Data," J Biophotonics, 2022. [doi:10.1002/jbio.202100330](https://doi.org/10.1002/jbio.202100330)

## Related Projects

Part of the [FlowRegSuite](https://github.com/FlowRegSuite) ecosystem:
- [Flow-Registration (MATLAB)](https://github.com/FlowRegSuite/flow_registration) - Original MATLAB implementation
- [napari-flowreg](https://github.com/FlowRegSuite/napari-flowreg) - Interactive visualization plugin

## Documentation

```{toctree}
:maxdepth: 2

installation
quickstart
api/index
user_guide/index
theory/index
changelog
```

## License

PyFlowReg is released under the CC BY-NC-SA 4.0 license.
