# <img src="img/flowreglogo.png" alt="FlowReg logo" height="64"> PyFlowReg

Python bindings & CLI for Flow-Registration - variational optical-flow motion correction for 2-photon (2P) microscopy videos and volumetric 3D scans.

Derived from the Flow-Registration toolbox for compensation and stabilization of multichannel microscopy videos. The original implementation spans MATLAB, Java (ImageJ/Fiji plugin), and C++. See the [publication](https://doi.org/10.1002/jbio.202100330) and the [project website](https://www.snnu.uni-saarland.de/flow-registration/) for method details and video results.

**Related projects**
- Original Flow-Registration repo: https://github.com/FlowRegSuite/flow_registration
- ImageJ/Fiji plugin: https://github.com/FlowRegSuite/flow_registration_IJ
- Napari plugin: https://github.com/FlowRegSuite/napari-flowreg
- MCP tools for LLM workflows: https://github.com/FlowRegSuite/pyflowreg-mcp

![Fig1](img/bg.jpg)


## Requirements

This code requires python 3.10 and cuda 11.8 for the gpu version. 

Initialize the environment with

```bash
conda create --name raft2p python=3.10
conda activate raft2p
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Installation via pip and conda

```bash
conda create --name pyflowreg python=3.10
pip install pyflowreg
```

## Getting started

This repository contains the demo scripts ```demos/jupiter.m``` and ```demos/jupiter_minimal_example.m``` which run out of the box and compensate the jitter in an amateur recording of a meteor impact on jupiter. The folder ```demos/examples``` contains examples that illustrate use cases of the toolbox and ```demos/reproduce_journal_results``` contains scripts that replicate the evaluations from our paper.

The plugin supports most of the commonly used file types such as HDF5, tiff stacks and matlab mat files. To run the motion compensation, the options need to be defined into a ```OF_options``` object such as


## Dataset

The dataset which we used for our evaluations is available as [2-Photon Movies with Motion Artifacts](https://www.datadryad.org).

## Citation

Details on the method and video results can be found [here](https://www.snnu.uni-saarland.de/flow-registration/).

If you use parts of this code or the plugin for your work, please cite

> P. Flotho, S. Nomura, M. Flotho, A. Keller and B. Kuhn, “Pyflowreg: A python package for high accuracy motion correction of 2P microscopy videos and 3D scans,” (in preparation), 2025. [doi:https://]()


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

