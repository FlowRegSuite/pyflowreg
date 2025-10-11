"""
I/O utilities for video file reading and writing.
"""

# Base classes
from ._base import VideoReader, VideoWriter

# Memory-based I/O
from ._arr import ArrayReader, ArrayWriter
from ._null import NullVideoWriter

# Factory functions
from .factory import get_video_file_reader, get_video_file_writer

# File format readers
from .hdf5 import HDF5FileReader, HDF5FileWriter
from .tiff import TIFFFileReader, TIFFFileWriter
from .mat import MATFileReader, MATFileWriter
from .multifile_wrappers import (
    MULTICHANNELFileReader,
    MULTIFILEFileWriter,
    SUBSETFileReader,
)

# Platform-specific (Windows only)
try:
    from .mdf import MDFFileReader
except ImportError:
    MDFFileReader = None

__all__ = [
    # Base classes
    "VideoReader",
    "VideoWriter",
    # Memory I/O
    "ArrayReader",
    "ArrayWriter",
    "NullVideoWriter",
    # Factory functions
    "get_video_file_reader",
    "get_video_file_writer",
    # File readers/writers
    "HDF5FileReader",
    "HDF5FileWriter",
    "TIFFFileReader",
    "TIFFFileWriter",
    "MATFileReader",
    "MATFileWriter",
    # Wrappers
    "MULTICHANNELFileReader",
    "MULTIFILEFileWriter",
    "SUBSETFileReader",
]

# Add MDFFileReader if available
if MDFFileReader is not None:
    __all__.append("MDFFileReader")
