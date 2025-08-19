# methods/__init__.py
from . import pyflowreg
from . import suite2p
# from . import normcorre  # Skip if CaImAn not installed
from . import antspyx
from . import elastix

__all__ = ["pyflowreg", "suite2p", "antspyx", "elastix"]