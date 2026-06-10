# Installation

## Requirements

PyFlowReg requires Python 3.10 or higher.

## Installation via pip

The recommended installation method is via pip:

```bash
pip install pyflowreg
```

### With Visualization Support

To install with visualization support (matplotlib, scikit-learn):

```bash
pip install pyflowreg[vis]
```

### Session Processing Support

Session processing support is included in the base install. The command-line
tools `pyflowreg-session` and `pyflowreg-z-align` are installed with
`pip install pyflowreg`.

### Complete Installation

For all optional features (visualization, cluster computing, and GPU flow backends):

```bash
pip install pyflowreg[vis,dask,gpu]
```

The available pip extras are:

- `vis`: matplotlib and scikit-learn for visualization utilities
- `dask`: Dask distributed and dask-jobqueue for cluster computing
- `gpu`: CuPy (CUDA 12; installed on Linux and Windows only) and PyTorch for the GPU flow backends (`flowreg_cuda`, `flowreg_torch`); see [Flow backends](user_guide/backends.md)
- `test`: pytest and pytest-cov for running the test suite
- `docs`: Sphinx and related tooling for building the documentation

### For Development

To install in development mode with test dependencies:

```bash
git clone https://github.com/FlowRegSuite/pyflowreg.git
cd pyflowreg
pip install -e .[test,vis,docs]
```

## Platform-Specific Notes

### Windows

MDF file support (Sutter file format) is only available on Windows and requires
`pywin32`. On Windows, `pywin32` is a regular (platform-conditional) dependency
of PyFlowReg and is installed automatically by `pip install pyflowreg`; no
extra step is needed.

## Using Mamba

Create a dedicated environment with mamba:

```bash
mamba create --name pyflowreg python=3.10
mamba activate pyflowreg
pip install pyflowreg
```

## Verifying Installation

Test your installation:

```python
import pyflowreg
from pyflowreg.motion_correction import OFOptions
from pyflowreg.session import SessionConfig
print(pyflowreg.__version__)
```

Verify command-line tools:

```bash
# Check session CLI is installed
pyflowreg-session --help

# Check z-alignment CLI is installed
pyflowreg-z-align --help
```
