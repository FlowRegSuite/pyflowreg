# Installation

## Requirements

PyFlowReg requires Python 3.10 or higher.

## Installation via pip

The recommended installation method is via pip:

```bash
pip install pyflowreg
```

### With Visualization Support

To install with full visualization support (matplotlib, scikit-learn):

```bash
pip install pyflowreg[vis]
```

### With Session Processing Support

For multi-session processing with configuration file support:

```bash
pip install pyflowreg[session]
```

This installs:
- `pyyaml` for YAML configuration files
- `Pillow` for PNG mask export
- Command-line tool: `pyflowreg-session`

### Complete Installation

For all features including session processing, visualization, and cluster computing:

```bash
pip install pyflowreg[session,vis,dask]
```

### For Development

To install in development mode with test dependencies:

```bash
git clone https://github.com/FlowRegSuite/pyflowreg.git
cd pyflowreg
pip install -e .[test,vis,docs]
```

## Platform-Specific Notes

### Windows

On Windows, MDF file support (Sutter file format) requires additional dependencies:

```bash
pip install pywin32
```

This is automatically installed when using `pip install pyflowreg` on Windows.

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
```
