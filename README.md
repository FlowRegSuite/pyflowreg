[![PyPI - Version](https://img.shields.io/pypi/v/pyflowreg)](https://pypi.org/project/pyflowreg/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyflowreg)](https://pypi.org/project/pyflowreg/)
[![PyPI - License](https://img.shields.io/pypi/l/pyflowreg)](LICENSE)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyflowreg)](https://pypistats.org/packages/pyflowreg)
[![GitHub Actions](https://github.com/FlowRegSuite/pyflowreg/actions/workflows/pypi-release.yml/badge.svg)](https://github.com/FlowRegSuite/pyflowreg/actions/workflows/pypi-release.yml)

## 🚧 Under Development

This project is still in an **alpha stage**. Expect rapid changes, incomplete features, and possible breaking updates between releases. 

- The API may evolve as we stabilize core functionality.  
- Documentation and examples are incomplete.  
- Feedback and bug reports are especially valuable at this stage.  

# <img src="https://raw.githubusercontent.com/FlowRegSuite/pyflowreg/refs/heads/main/img/flowreglogo.png" alt="FlowReg logo" height="64"> PyFlowReg

Python bindings for Flow-Registration - variational optical-flow motion correction for 2-photon (2P) microscopy videos and volumetric 3D scans.

Derived from the Flow-Registration toolbox for compensation and stabilization of multichannel microscopy videos. The original implementation spans MATLAB, Java (ImageJ/Fiji plugin), and C++. See the [publication](https://doi.org/10.1002/jbio.202100330) and the [project website](https://www.snnu.uni-saarland.de/flow-registration/) for method details and video results.

**Related projects**
- Original Flow-Registration repo: https://github.com/FlowRegSuite/flow_registration
- ImageJ/Fiji plugin: https://github.com/FlowRegSuite/flow_registration_IJ
- Napari plugin: https://github.com/FlowRegSuite/napari-flowreg


![Fig1](https://raw.githubusercontent.com/FlowRegSuite/pyflowreg/refs/heads/main/img/bg.jpg)


## Requirements

This code requires python 3.10 or higher.

Initialize the environment with

```bash
conda create --name pyflowreg python=3.10
conda activate pyflowreg
pip install -r requirements.txt
```

or on windows

```bash
pip install -r requirements.txt
```

to enable Sutter MDF file support.


## Installation via pip and conda

```bash
conda create --name pyflowreg python=3.10
pip install pyflowreg
```

To install the project with full visualization support, you can install the `napari` plugin:

```bash
pip install pyflowreg[vis]
```

## Getting started

This repository contains demo scripts under ```experiments``` and 
demo notebooks under ```notebooks```. The demos with the jupiter sequence should run out of the box.

The plugin supports most of the commonly used file types such as HDF5, tiff stacks and matlab mat files. To run the motion compensation, the options need to be defined into a ```OF_options``` object.

The python version of Flow-Registration aims at full MATLAB compatibility, any missing functionality should be reported as an issue. The API is designed to be similar to the original MATLAB code, with some adjustments for Python conventions.


## Dataset

The dataset which we used for our evaluations is available as [2-Photon Movies with Motion Artifacts](https://drive.google.com/drive/folders/1fPdzQo5SiA-62k4eHF0ZaKJDt1vmTVed?usp=sharing).

## Citation

Details on the original method and video results can be found [here](https://www.snnu.uni-saarland.de/flow-registration/).

If you use parts of this code or the plugin for your work, please cite

> “Pyflowreg,” (in preparation), 2025.


and for Flow-Registration

> P. Flotho, S. Nomura, B. Kuhn and D. J. Strauss, “Software for Non-Parametric Image Registration of 2-Photon Imaging Data,” J Biophotonics, 2022. [doi:https://doi.org/10.1002/jbio.202100330](https://doi.org/10.1002/jbio.202100330)

BibTeX entry
```
@article{flotea2022a,
    author = {Flotho, P. and Nomura, S. and Kuhn, B. and Strauss, D. J.},
    title = {Software for Non-Parametric Image Registration of 2-Photon Imaging Data},
    year = {2022},
  journal = {J Biophotonics},
  doi = {https://doi.org/10.1002/jbio.202100330}
}
```

