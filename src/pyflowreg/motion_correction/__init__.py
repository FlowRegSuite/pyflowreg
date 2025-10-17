"""
Motion Correction and Analysis for 2-Photon Microscopy
=======================================================

This module provides high-level APIs for motion correction and motion analysis
of microscopy videos using variational optical flow. Dense motion information
is explicitly computed and returned, enabling subsequent motion visualization
or quantitative analysis in addition to motion-corrected output.

The module includes both array-based and file-based processing with support for
multi-channel data, reference frame alignment, and parallel execution.

Main Functions
--------------
compensate_arr
    Motion correction for in-memory video arrays
compensate_recording
    Motion correction for video files with parallel processing

Classes
-------
OFOptions
    Configuration class for all optical flow and processing parameters
BatchMotionCorrector
    Batch processor for file-based motion correction with parallel execution
FlowRegLive
    Real-time motion correction with adaptive reference updating

Quick Start
-----------
>>> from pyflowreg.motion_correction import compensate_arr, OFOptions
>>> options = OFOptions(quality_setting="balanced")
>>> registered, flow = compensate_arr(video, reference, options)

See Also
--------
pyflowreg.core : Low-level optical flow computation
pyflowreg.util : I/O, visualization, and image processing utilities
"""

from pyflowreg.motion_correction.compensate_arr import compensate_arr
from pyflowreg.motion_correction.compensate_recording import (
    compensate_recording,
    BatchMotionCorrector,
)
from pyflowreg.motion_correction.OF_options import (
    OFOptions,
    QualitySetting,
    OutputFormat,
    ChannelNormalization,
)
from pyflowreg.motion_correction.flow_reg_live import FlowRegLive

__all__ = [
    "compensate_arr",
    "compensate_recording",
    "BatchMotionCorrector",
    "OFOptions",
    "QualitySetting",
    "OutputFormat",
    "ChannelNormalization",
    "FlowRegLive",
]
