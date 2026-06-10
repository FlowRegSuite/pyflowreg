"""
Z-alignment pipeline for depth-shift correction.

The ``z_align`` module implements a stage-based workflow that mirrors the
MATLAB prototype used for z-shift estimation/correction:

1. Build/load a reference volume
2. Estimate per-pixel z shifts and optionally write z-corrected output
3. Optionally simulate a z-shift-only recording from the estimated z shifts
"""

from pyflowreg.z_align.config import ZAlignConfig
from pyflowreg.z_align.pipeline import (
    run_recording_prealignment,
    run_stage1,
    run_stage2,
    run_stage3,
    run_all_stages,
)

__all__ = [
    "ZAlignConfig",
    "run_recording_prealignment",
    "run_stage1",
    "run_stage2",
    "run_stage3",
    "run_all_stages",
]
