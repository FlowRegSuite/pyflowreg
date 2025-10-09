import gc
import os
import time
from typing import Union, List

import numpy as np

from pyflowreg.util.io._base import VideoReader

# =============================================================================
# IMPORTANT WARNING:
# This module requires the 'pywin32' library and can ONLY run on Windows.
# It interacts with the 'MCSX.Data' COM server, which must be installed
# on the system (e.g., by installing the original MDF software).
# A pure Python, cross-platform solution is not possible without a dedicated
# library to parse this specific MDF format.
# =============================================================================

try:
    import win32com.client

    MDF_SUPPORTED = True
except ImportError:
    MDF_SUPPORTED = False


class MDFFileReader(VideoReader):
    """
    MDF file reader for Windows using MCSX.Data COM interface.

    Note: MDF files use 1-based indexing internally (MATLAB heritage).
    This reader transparently converts between 0-based Python indexing
    and 1-based MDF indexing.
    """

    def __init__(
        self, file_path: str, buffer_size: int = 500, bin_size: int = 1, **kwargs
    ):
        """
        Initialize MDF reader.

        Args:
            file_path: Path to .mdf file
            buffer_size: Number of frames per batch
            bin_size: Temporal binning factor
            channel_idx: Optional list of channels to read (1-based)
        """
        if not MDF_SUPPORTED:
            raise NotImplementedError(
                "MDF file reading requires Windows and 'pywin32' library"
            )

        super().__init__()

        self.file_path = file_path
        self.buffer_size = buffer_size
        self.bin_size = bin_size
        self.mfile = None

        # MDF-specific options
        self.channel_idx = kwargs.get("channel_idx", None)  # Will be set in _initialize
        self._out_of_bound_warning = True

        # Validate file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"MDF file not found: {file_path}")

    def _initialize(self):
        """Open MDF file and read metadata."""
        try:
            self.mfile = win32com.client.Dispatch("MCSX.Data")
            if self.mfile.OpenMCSFile(self.file_path):
                raise ConnectionAbortedError(
                    "Failed to open MDF file. Only one instance can be opened at once. "
                    "Close other MDF viewers and clear any other instances."
                )
        except Exception as e:
            raise ConnectionError(f"Could not connect to MCSX.Data COM server: {e}")

        # Read core metadata
        self.frame_count = int(self.mfile.ReadParameter("Frame Count"))
        self.height = int(self.mfile.ReadParameter("Frame Height"))
        self.width = int(self.mfile.ReadParameter("Frame Width"))

        # Determine bit depth and dtype
        bit_depth_str = self.mfile.ReadParameter("Frame Bit Depth").split("-")[0]
        bit_depth = int(bit_depth_str)
        self.dtype = self._get_numpy_dtype(bit_depth)

        # Detect available channels (MDF uses 0-based channel indices in metadata)
        available_channels = []
        for i in range(3):  # Check channels 0, 1, 2 in metadata
            if self.mfile.ReadParameter(f"Scanning Ch {i} Name"):
                available_channels.append(i + 1)  # Convert to 1-based for ReadFrame

        # Set channels to read
        if self.channel_idx is None:
            self.channel_idx = available_channels
        else:
            # Validate requested channels
            for ch in self.channel_idx:
                if ch not in available_channels:
                    raise ValueError(
                        f"Channel {ch} not available. Available: {available_channels}"
                    )

        self.n_channels = len(self.channel_idx)

    def _get_numpy_dtype(self, bit_depth: int) -> np.dtype:
        """Map bit depth to numpy dtype."""
        if bit_depth <= 8:
            return np.uint8
        elif bit_depth <= 16:
            return np.uint16
        else:
            return np.float64

    def _clean_frame_data(self, raw_data: tuple, frame_idx: int) -> np.ndarray:
        """
        Clean and validate frame data from COM interface.

        Args:
            raw_data: Raw data tuple from COM interface
            frame_idx: Frame index (0-based) for error messages

        Returns:
            Cleaned numpy array with proper dtype
        """
        # Convert to numpy array
        temp_array = np.array(raw_data)

        # Get valid bounds for target dtype
        if np.issubdtype(self.dtype, np.integer):
            dtype_info = np.iinfo(self.dtype)
            min_val, max_val = dtype_info.min, dtype_info.max
        else:
            # For float types, no clamping needed
            return temp_array.astype(self.dtype)

        # Check for out-of-bounds values
        has_negatives = np.any(temp_array < min_val)
        has_overflow = np.any(temp_array > max_val)

        # Warn once about out-of-bounds values
        if self._out_of_bound_warning and (has_negatives or has_overflow):
            if np.issubdtype(self.dtype, np.unsignedinteger) and has_negatives:
                print(
                    f"Warning: Negative values in frame {frame_idx}, "
                    f"clamping to 0 for dtype {self.dtype}"
                )
            if has_overflow:
                print(
                    f"Warning: Values exceeding {max_val} in frame {frame_idx}, "
                    f"clamping to maximum for dtype {self.dtype}"
                )
            self._out_of_bound_warning = False

        # Clamp and convert
        np.clip(temp_array, min_val, max_val, out=temp_array)
        return temp_array.astype(self.dtype)

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """
        Read raw frames from MDF file.

        Args:
            frame_indices: 0-based indices (slice or list)

        Returns:
            Array with shape (T, H, W, C)
        """
        # Convert slice to list of indices
        if isinstance(frame_indices, slice):
            start, stop, step = frame_indices.indices(self.frame_count)
            frame_indices = list(range(start, stop, step))

        n_frames = len(frame_indices)
        if n_frames == 0:
            return np.empty(
                (0, self.height, self.width, self.n_channels), dtype=self.dtype
            )

        # Allocate output array
        output = np.zeros(
            (n_frames, self.height, self.width, self.n_channels), dtype=self.dtype
        )

        # Read each frame
        for i, frame_idx in enumerate(frame_indices):
            # Convert to 1-based indexing for MDF
            mdf_frame_idx = frame_idx + 1

            # Validate frame index
            if mdf_frame_idx < 1 or mdf_frame_idx > self.frame_count:
                raise IndexError(
                    f"Frame index {frame_idx} out of range [0, {self.frame_count})"
                )

            # Read each channel for this frame
            for ch_idx, channel_num in enumerate(self.channel_idx):
                # ReadFrame expects (channel_number, frame_number) both 1-based
                raw_data = self.mfile.ReadFrame(channel_num, mdf_frame_idx)

                if raw_data is None:
                    raise IOError(
                        f"Failed to read frame {frame_idx} (MDF frame {mdf_frame_idx}) "
                        f"from channel {channel_num}. File may be locked or corrupted."
                    )

                # Clean and transpose data
                # MDF returns data in column-major order (Fortran-style)
                # so we need to transpose to get (H, W)
                cleaned_data = self._clean_frame_data(raw_data, frame_idx)
                output[i, :, :, ch_idx] = cleaned_data.T

        return output

    def close(self):
        """Release COM object and clean up."""
        if self.mfile:
            self.mfile = None
            gc.collect()  # Force garbage collection for COM cleanup

    def reset_connection(self):
        """
        Reset the MDF COM connection if it becomes unresponsive.
        Useful when the COM server gets stuck.
        """
        print("Resetting MDF connection...")

        # Release current connection
        self.mfile = None
        gc.collect()
        time.sleep(0.1)  # Give OS time to release file locks

        try:
            # Re-establish connection
            self.mfile = win32com.client.Dispatch("MCSX.Data")
            if self.mfile.OpenMCSFile(self.file_path):
                raise ConnectionAbortedError("Failed to re-open MDF file")

            # Reset state
            self.current_frame = 0
            print("Connection successfully reset")

        except Exception as e:
            raise ConnectionError(f"Could not re-establish connection: {e}")

    def get_metadata(self) -> dict:
        """
        Get comprehensive metadata from MDF file.

        Returns:
            Dictionary with file metadata
        """
        self._ensure_initialized()

        # Helper to parse parameters with units
        def parse_param(value: str, unit: str = "", dtype=float):
            try:
                if value:
                    clean = value.replace(",", ".").replace(unit, "").strip()
                    return dtype(clean)
            except (ValueError, AttributeError, TypeError):
                # Silently return None for missing/unparseable metadata (expected behavior)
                return None

        # Read various parameters
        microns_per_pixel = parse_param(
            self.mfile.ReadParameter("Microns per Pixel"), "µm"
        )
        magnification = parse_param(self.mfile.ReadParameter("Magnification"), "x")
        frame_duration_s = parse_param(
            self.mfile.ReadParameter("Frame Duration (s)"), "s"
        )
        frame_interval_ms = parse_param(
            self.mfile.ReadParameter("Frame Interval (ms)"), "ms"
        )

        # Calculate time step
        dt = (frame_duration_s or 0) + (frame_interval_ms or 0) / 1000.0
        if dt == 0:
            dt = 1 / 30.91  # Default from MATLAB code
            print("Warning: Could not read frame timing, using default 30.91 Hz")

        # Get channel names
        channel_names = []
        for ch in self.channel_idx:
            # Channel index in metadata is 0-based
            name = self.mfile.ReadParameter(f"Scanning Ch {ch - 1} Name")
            channel_names.append(name or f"Channel {ch}")

        return {
            "file_name": os.path.basename(self.file_path),
            "frame_count": self.frame_count,
            "shape": self.shape,
            "unbinned_shape": self.unbinned_shape,
            "channels": channel_names,
            "dt_seconds": dt * self.bin_size,
            "pixel_size_um": microns_per_pixel or 1.0,
            "magnification": magnification or 1.0,
            "dtype": str(self.dtype),
            "created_on": self.mfile.ReadParameter("Created On"),
        }


if __name__ == "__main__":
    filename = "D:\\2025_OIST\\Shinobu\\RFPonly\\190403_001.MDF"
    reader = MDFFileReader(filename, buffer_size=10, bin_size=50)
    # reader.reset_connection()
    vid = reader.read_batch()
    import cv2

    cv2.imshow(
        "Frame",
        cv2.normalize(vid[0, :, :, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
    )
    cv2.imshow(
        "Frame",
        cv2.normalize(
            reader[0:5][0, :, :, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        ),
    )
    cv2.waitKey()
