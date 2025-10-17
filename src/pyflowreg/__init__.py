"""
PyFlowReg: Variational Optical Flow for 2-Photon Microscopy
============================================================

PyFlowReg provides high-accuracy motion correction for 2-photon microscopy
videos and volumetric 3D scans using variational optical flow techniques.

This is a Python port of the Flow-Registration MATLAB toolbox, maintaining
full compatibility while adding modern Python features and optimizations.

Quick Start
-----------
>>> import pyflowreg
>>> from pyflowreg.motion_correction import compensate_arr, OFOptions
>>>
>>> # Set up options
>>> options = OFOptions(quality_setting="balanced")
>>>
>>> # Register video (array-based)
>>> registered, flow = compensate_arr(video, reference, options)

Main Modules
------------
core
    Optical flow computation engine (get_displacement)
motion_correction
    High-level APIs (compensate_arr, compensate_recording, OFOptions)
util
    I/O, visualization, and image processing utilities

See Also
--------
Flow-Registration MATLAB : https://github.com/FlowRegSuite/flow_registration
Napari Plugin : https://github.com/FlowRegSuite/napari-flowreg
Publication : https://doi.org/10.1002/jbio.202100330

Notes
-----
This Python implementation maintains full algorithmic compatibility with the
MATLAB version. Any differences in behavior should be reported as issues.
"""

from pyflowreg.core.optical_flow import get_displacement as get_displacement

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
