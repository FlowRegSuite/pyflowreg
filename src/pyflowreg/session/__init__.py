"""
Session-level multi-recording processing module.

This module provides functionality for processing multiple recording sequences
in a session, including:
- Per-recording motion correction (Stage 1)
- Inter-sequence alignment (Stage 2)
- Valid mask computation and alignment (Stage 3)

Mirrors MATLAB session processing workflow from FlowRegSuite.
"""

from pyflowreg.session.config import SessionConfig

__all__ = ["SessionConfig"]
