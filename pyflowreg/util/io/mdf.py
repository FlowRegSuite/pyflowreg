import numpy as np
from datetime import datetime
from typing import Optional, Any
from pyflowreg.util.io.base import VideoReader
import time
import gc

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
    A class for reading data from an MDF (Multi-Dimensional Format) file.

    This class is a Python port of the MATLAB MDF_file_reader and relies on the
    MCSX.Data COM server for file access.
    """

    def __init__(self, input_file: str, buffer_size: int = 500, bin_size: int = 1, **kwargs):
        """
        Initializes the MDF file reader.

        Args:
            input_file (str): Path to the .mdf file.
            buffer_size (int): Number of frames to read in a batch.
            bin_size (int): Binning factor (Note: binning logic is not implemented here).
            **kwargs: Optional arguments, e.g., 'channel_idx' to specify channels.
        """
        if not MDF_SUPPORTED:
            raise NotImplementedError(
                "MDF file reading is not supported on this platform. "
                "It requires Windows and the 'pywin32' library."
            )

        super().__init__()
        self.file_name = input_file
        self.buffer_size = buffer_size
        self.bin_size = bin_size  # Note: The bin_buffer method needs to be implemented.
        self._out_of_bound_warning = True

        try:
            self.mfile = win32com.client.Dispatch('MCSX.Data')
            if self.mfile.OpenMCSFile(input_file):
                raise ConnectionAbortedError(
                    "Failed to open MDF file. Only one instance can be opened at once. "
                    "Close other MDF viewers and clear any other instances."
                )
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to MCSX.Data COM server. Is the required software installed? Error: {e}")

        # --- Read metadata from the file ---
        self.frame_count = int(self.mfile.ReadParameter('Frame Count'))
        self.height = int(self.mfile.ReadParameter('Frame Height'))
        self.width = int(self.mfile.ReadParameter('Frame Width'))
        bit_depth_str = self.mfile.ReadParameter('Frame Bit Depth').split('-')[0]
        self.bit_depth = int(bit_depth_str)
        self.dtype = self._get_numpy_dtype(self.bit_depth)

        # --- Determine available channels ---
        available_channels = []
        for i in range(3):  # Check channels 0, 1, 2
            if self.mfile.ReadParameter(f'Scanning Ch {i} Name'):
                available_channels.append(i + 1)  # Use 1-based index like MATLAB code

        self.channel_idx = kwargs.get('channel_idx', available_channels)
        self.n_channels = min(len(available_channels), len(self.channel_idx))

    def _get_numpy_dtype(self, bit_depth: int) -> np.dtype:
        """Maps bit depth to a numpy dtype."""
        if bit_depth <= 8:
            return np.uint8
        elif bit_depth <= 16:
            return np.uint16
        else:
            return np.double

    def _bin_buffer(self, buffer: np.ndarray) -> np.ndarray:
        """
        Placeholder for the binning logic. The original MATLAB code refers to
        this method from a parent class, which was not provided.
        """
        if self.bin_size > 1:
            # You would need to implement the frame binning logic here.
            # For now, it just returns the buffer as is.
            print(f"Warning: Frame binning (bin_size={self.bin_size}) is not implemented.")
        return buffer

    def _clean_frame_data(self, raw_frame_data: tuple, frame_index: int) -> np.ndarray:
        """
        Checks data against the target dtype's bounds, warns once, and clamps.

        This function inspects the raw data and ensures it fits within the valid
        range of self.dtype (e.g., 0-65535 for uint16).
        """
        # Use a high-precision signed integer for temporary calculations to avoid overflow.
        temp_array = np.array(raw_frame_data)

        # Get the valid min and max values for the target data type (e.g., uint16).
        dtype_info = np.iinfo(self.dtype)
        min_bound, max_bound = dtype_info.min, dtype_info.max

        # Check for values that are out of the target data type's bounds.
        negatives_exist = np.any(temp_array < min_bound)
        positives_exist = np.any(temp_array > max_bound)

        # If out-of-bounds values are found and we haven't warned yet, issue warnings.
        if self._out_of_bound_warning and (negatives_exist or positives_exist):
            # Specifically check if the target is unsigned to provide the right context.
            if np.issubdtype(self.dtype, np.unsignedinteger):
                if negatives_exist:
                    print(f"Warning: Negative values detected in frame {frame_index}. "
                          f"Clamping to the target minimum of {min_bound} for '{self.dtype}'.")

            # This warning applies to both signed and unsigned out-of-bounds values.
            if positives_exist:
                print(f"Warning: Values exceeding target maximum detected in frame {frame_index}. "
                      f"Clamping to the target maximum of {max_bound} for '{self.dtype}'.")

            # Set the flag to False to prevent future warnings.
            self._out_of_bound_warning = False

        # Clamp the array to the valid range of the target dtype and cast it.
        np.clip(temp_array, min_bound, max_bound, out=temp_array)

        return temp_array.astype(self.dtype)

    def read_batch(self) -> Optional[np.ndarray]:
        """Reads the next batch of frames."""
        if self.current_frame >= self.frame_count:
            return None

        n_frames_to_read = min(self.buffer_size * self.bin_size, self.frame_count - self.current_frame)

        # Pre-allocate numpy array: (height, width, channels, frames)
        buffer = np.zeros((self.height, self.width, self.n_channels, n_frames_to_read), dtype=self.dtype)

        for i in range(n_frames_to_read):
            self.current_frame += 1
            for j in range(self.n_channels):
                raw_data = self.mfile.ReadFrame(self.channel_idx[j], self.current_frame)
                if raw_data is None:
                    raise IOError(
                        f"Failed to read frame {self.current_frame} from channel {self.channel_idx[j]}. "
                        "The file might be locked by another process, the index might be out of bounds, "
                        "or the file could be corrupted. Try restarting the Python kernel."
                    )
                cleaned_data = self._clean_frame_data(raw_data, self.current_frame)
                buffer[:, :, j, i] = cleaned_data.T

        return self._bin_buffer(buffer)

    def read_frames(self, frame_indices: list[int]) -> Optional[np.ndarray]:
        """Reads specific frames by their 1-based indices."""
        if not all(1 <= idx <= self.frame_count for idx in frame_indices):
            raise ValueError("All frame indices must be within the valid frame count.")

        n_elements = len(frame_indices)
        buffer = np.zeros((self.height, self.width, self.n_channels, n_elements), dtype=self.dtype)

        for i, frame_idx in enumerate(frame_indices):
            for j in range(self.n_channels):
                frame_data = self.mfile.ReadFrame(self.channel_idx[j], frame_idx)
                buffer[:, :, j, i] = np.array(frame_data, dtype=self.dtype).T

        return self._bin_buffer(buffer)

    def get_tseries_metadata(self) -> dict[str, Any]:
        """Extracts and formats metadata into a dictionary."""

        def parse_param(value: str, unit: str, to_type=float) -> Any:
            try:
                # MATLAB code replaces comma with period for decimals
                clean_val = value.replace(',', '.').replace(unit, '').strip()
                return to_type(clean_val)
            except (ValueError, AttributeError):
                return None

        microns_per_pixel = parse_param(self.mfile.ReadParameter('Microns per Pixel'), 'µm')
        magnification = parse_param(self.mfile.ReadParameter('Magnification'), 'x')
        frame_duration_s = parse_param(self.mfile.ReadParameter('Frame Duration (s)'), 's')
        frame_interval_ms = parse_param(self.mfile.ReadParameter('Frame Interval (ms)'), 'ms')

        dt = (frame_duration_s or 0) + (frame_interval_ms or 0) / 1000.0
        if dt == 0:
            dt = 1 / 30.91  # Default fallback from MATLAB code
            print("Warning: Could not read frame duration/interval. Using default dt.")

        channel_names = [
            self.mfile.ReadParameter(f'Scanning Ch {i - 1} Name') for i in self.channel_idx
        ]

        start_time_str = self.mfile.ReadParameter('Created On')
        try:
            # Python format string for 'dddd, MMMM d, yyyy h:mm:ss a'
            start_time = datetime.strptime(start_time_str, '%A, %B %d, %Y %I:%M:%S %p')
        except ValueError:
            start_time = None

        return {
            "name": self.file_name.split('\\')[-1].split('.')[0],
            "frame_count": self.frame_count,
            "channel_names": channel_names,
            "channels": self.n_channels,
            "img_dim": (self.height, self.width),
            "dt_s": dt * self.bin_size,
            "dx_um": microns_per_pixel or 1.0,
            "dy_um": microns_per_pixel or 1.0,
            "zoom": magnification or 1.0,
            "start_time": start_time
        }

    def has_batch(self) -> bool:
        """Returns True if there are more frames to be read."""
        return self.current_frame < self.frame_count

    def reset(self):
        """Resets the reader to the beginning of the file."""
        self.current_frame = 0

    def close(self):
        """Releases the COM object."""
        if self.mfile:
            self.mfile = None  # Release the COM object
            print("MDF file reader closed.")

    def reset_connection(self):
        """
        Releases the current COM object and establishes a new, clean connection.
        """
        print("Attempting to reset the MDF file connection...")

        # 1. Release the Python reference to the object.
        self.mfile = None

        # 2. Force Python's garbage collector to run. This is crucial for COM
        #    objects to ensure the underlying Windows resource is released promptly.
        gc.collect()

        # 3. Wait for a brief moment. This gives the OS time to fully
        #    release any file locks before you try to grab it again.
        time.sleep(0.1)

        # 4. Re-instantiate a brand new COM object.
        try:
            self.mfile = win32com.client.Dispatch('MCSX.Data')

            # 5. Re-open the file with the new object.
            if self.mfile.OpenMCSFile(self.file_name):
                raise ConnectionAbortedError("Failed to re-open MDF file after reset.")

            # 6. Reset the reader's internal state.
            self.current_frame = 0
            print("Connection successfully reset.")

        except Exception as e:
            raise ConnectionError(f"Could not re-establish connection to MCSX.Data. Error: {e}")


if __name__ == "__main__":
    filename = "E:\\data\\2025_OIST\\RFPonly\\190403_001.MDF"
    reader = MDFFileReader(filename, buffer_size=10, bin_size=1)
    # reader.reset_connection()
    vid = reader.read_batch()
    import cv2
    cv2.imshow("Frame", cv2.normalize(
        vid[:, :, 0, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    ))
    cv2.waitKey()
