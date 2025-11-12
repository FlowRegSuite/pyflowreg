import os
import warnings
from typing import Union, List
import numpy as np

try:
    import tifffile

    TIFF_SUPPORTED = True
except ImportError:
    TIFF_SUPPORTED = False
    warnings.warn("tifffile not installed. TIFF support unavailable.")

from pyflowreg.util.io._base import VideoReader, VideoWriter
from pyflowreg.util.io._scanimage import parse_scanimage_metadata


class TIFFFileReader(VideoReader):
    """
    TIFF stack file reader with support for multi-page and multi-channel formats.

    Supports:
    - Multi-page TIFF stacks (standard format)
    - Single-page multi-sample TIFFs (channels as samples per pixel)
    - Deinterleaved reading for formats like Suite2p
    - Memory-mapped reading for large files
    - Various data types (uint8/16/32/64, int32/64, float32/64)
    """

    def __init__(
        self, file_path: str, buffer_size: int = 500, bin_size: int = 1, **kwargs
    ):
        """
        Initialize TIFF reader.

        Args:
            file_path: Path to TIFF file
            buffer_size: Number of frames per batch
            bin_size: Temporal binning factor
            deinterleave: Channel deinterleaving factor (1=none, >1 for interleaved formats)
            use_memmap: Use memory mapping for large files (default: True)
        """
        if not TIFF_SUPPORTED:
            raise ImportError("tifffile library required for TIFF support")

        # Initialize parent classes
        VideoReader.__init__(self)

        self.file_path = file_path
        self.buffer_size = buffer_size
        self.bin_size = bin_size

        # TIFF-specific options
        self.deinterleave = kwargs.get("deinterleave", 1)
        self.use_memmap = kwargs.get("use_memmap", True)

        # Internal state
        self._tiff_file = None
        self._tiff_series = None
        self._sample_mode = False
        self._page_indices = None

        # ScanImage metadata
        self._scanimage_metadata = None
        self._is_scanimage = False
        self._z_stack_info = None

        # Validate file
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"TIFF file not found: {file_path}")

    def _initialize(self):
        """Open TIFF file and read metadata."""
        try:
            # Open with tifffile
            self._tiff_file = tifffile.TiffFile(self.file_path)

            # Check if it's a ScanImage file
            self._check_scanimage_metadata()

            # Check if it's an ImageJ or OME-TIFF with series
            if len(self._tiff_file.series) > 0:
                self._tiff_series = self._tiff_file.series[0]
                self._setup_from_series()
            else:
                # Fallback to page-based reading
                self._setup_from_pages()

        except Exception as e:
            raise IOError(f"Failed to open TIFF file: {e}")

    def _check_scanimage_metadata(self):
        """Check if this is a ScanImage TIFF and parse metadata if so."""
        try:
            # First check if tifffile directly identifies it as ScanImage
            if hasattr(self._tiff_file, "is_scanimage"):
                self._is_scanimage = self._tiff_file.is_scanimage
            else:
                self._is_scanimage = False

            # If it's ScanImage, parse detailed metadata
            if self._is_scanimage:
                # Try to get metadata from tifffile first
                if hasattr(self._tiff_file, "scanimage_metadata"):
                    si_metadata = self._tiff_file.scanimage_metadata

                    # Parse framesPerSlice and numSlices
                    if "SI" in si_metadata and "hStackManager" in si_metadata["SI"]:
                        stack = si_metadata["SI"]["hStackManager"]
                        slices = stack.get("actualNumSlices", stack.get("numSlices", 1))
                        frames_per_slice = stack.get("framesPerSlice", 1)
                        volumes = stack.get(
                            "actualNumVolumes", stack.get("numVolumes", 1)
                        )
                        z_step = stack.get(
                            "stackZStepSize", stack.get("actualStackZStepSize", None)
                        )

                        # Parse channel information
                        channels_saved = 1
                        if "hChannels" in si_metadata["SI"]:
                            chan_info = si_metadata["SI"]["hChannels"]
                            if "channelSave" in chan_info:
                                channels_saved = len(chan_info["channelSave"])

                        # Check if we need to auto-deinterleave for ScanImage
                        # ScanImage often stores channels as interleaved pages
                        if channels_saved > 1 and self.deinterleave == 1:
                            # Check if page count suggests interleaved channels
                            expected_pages = (
                                slices * frames_per_slice * volumes * channels_saved
                            )
                            actual_pages = len(self._tiff_file.pages)

                            if actual_pages == expected_pages:
                                print(
                                    f"ScanImage channel interleaving detected: auto-setting deinterleave={channels_saved}"
                                )
                                self.deinterleave = channels_saved

                        if slices > 1 or frames_per_slice > 1:
                            self._z_stack_info = {
                                "volumes": volumes,
                                "slices_per_volume": slices,
                                "frames_per_slice": frames_per_slice,
                                "total_frames_flattened": slices
                                * frames_per_slice
                                * volumes,
                                "z_step": z_step,
                                "interpretation": "z_stack",
                                "channels_saved": channels_saved,
                            }
                            print("ScanImage Z-stack detected from metadata:")
                            print(
                                f"  {slices} slices × {frames_per_slice} frames/slice × {volumes} volume(s)"
                            )
                            print(
                                f"  = {slices * frames_per_slice * volumes} total frames"
                            )
                            print(f"  Channels: {channels_saved}")
                            if z_step:
                                print(f"  Z step: {z_step} µm")
                else:
                    # Fallback to parsing from file
                    self._scanimage_metadata = parse_scanimage_metadata(self.file_path)

                    if self._scanimage_metadata.get("is_scanimage"):
                        slices = self._scanimage_metadata.get("slices_per_volume", 1)
                        volumes = self._scanimage_metadata.get("volumes", 1)

                        if slices > 1 or volumes > 1:
                            self._z_stack_info = {
                                "volumes": volumes,
                                "slices_per_volume": slices,
                                "total_frames_flattened": volumes * slices,
                                "z_step": self._scanimage_metadata.get("z_step"),
                                "interpretation": self._scanimage_metadata.get(
                                    "interpretation", "z_stack"
                                ),
                            }
                            print(
                                f"ScanImage Z-stack detected: {volumes} volume(s), {slices} slice(s) per volume"
                            )
                            print(
                                f"  Treating as {volumes * slices} individual frames for motion correction"
                            )

        except Exception as e:
            # If parsing fails, treat as regular TIFF
            print(f"Warning: Error parsing ScanImage metadata: {e}")
            self._is_scanimage = False
            self._scanimage_metadata = None

    def _setup_from_series(self):
        """Setup reader from tifffile series (standard multi-page format)."""
        shape = self._tiff_series.shape
        axes = self._tiff_series.axes

        # Get spatial dimensions first
        y_idx = axes.index("Y") if "Y" in axes else 0
        x_idx = axes.index("X") if "X" in axes else 1
        self.height = shape[y_idx]
        self.width = shape[x_idx]

        # Get channels
        if "C" in axes:
            c_idx = axes.index("C")
            self.n_channels = shape[c_idx]
        elif "S" in axes:
            # Samples as channels
            s_idx = axes.index("S")
            self.n_channels = shape[s_idx]
        else:
            self.n_channels = 1

        # For ScanImage files with channels, auto-enable deinterleaving
        # ScanImage stores multi-channel data as interleaved pages
        if self._is_scanimage and "C" in axes and shape[axes.index("C")] > 1:
            if self.deinterleave == 1:  # Only auto-set if not manually specified
                self.deinterleave = shape[axes.index("C")]
                print(
                    f"ScanImage multi-channel detected: auto-setting deinterleave={self.deinterleave}"
                )
                # When deinterleaving, we treat pages as interleaved channels
                self.n_channels = self.deinterleave

        # Handle frame count based on whether this is ScanImage with Z-stacks
        if self._is_scanimage and self._z_stack_info:
            # For ScanImage Z-stacks, flatten volumes * slices into frames
            self.frame_count = self._z_stack_info["total_frames_flattened"]
        else:
            # Standard TIFF handling - but check for ScanImage-style ZTCYX
            # Common patterns: 'TYX', 'TCYX', 'YXS', 'ZCYX', 'ZTCYX'
            if "Z" in axes and "T" in axes:
                # Both Z and T present (like ScanImage ZTCYX)
                z_idx = axes.index("Z")
                t_idx = axes.index("T")

                # For ScanImage files, even without explicit metadata,
                # we should flatten Z*T as total frames
                if self._is_scanimage:
                    self.frame_count = shape[z_idx] * shape[t_idx]
                    print(
                        f"ScanImage ZTCYX detected: {shape[z_idx]} Z-slices × {shape[t_idx]} frames/slice = {self.frame_count} total frames"
                    )
                else:
                    # For non-ScanImage, T is primary time dimension
                    self.frame_count = shape[t_idx]
            elif "T" in axes:
                # Time series only
                time_idx = axes.index("T")
                self.frame_count = shape[time_idx]
            elif "Z" in axes:
                # Z-stack only, treated as time
                z_idx = axes.index("Z")
                self.frame_count = shape[z_idx]
            else:
                # Single frame or special format
                self.frame_count = len(self._tiff_file.pages)

        # Apply deinterleaving to frame count if needed
        if self.deinterleave > 1:
            self.frame_count = self.frame_count // self.deinterleave

        # Get data type
        self.dtype = self._tiff_series.dtype

    def _setup_from_pages(self):
        """Setup reader from individual pages (fallback method)."""
        pages = self._tiff_file.pages
        first_page = pages[0]

        # Check if single page with multiple samples (sample mode)
        if len(pages) == 1 and hasattr(first_page, "samplesperpixel"):
            samples = first_page.samplesperpixel
            if samples > 1:
                self._sample_mode = True
                self.frame_count = samples
                self.n_channels = 1
                self.height = first_page.imagelength
                self.width = first_page.imagewidth
            else:
                # Single page, single sample
                self.frame_count = 1
                self.n_channels = 1
                self.height = first_page.imagelength
                self.width = first_page.imagewidth
        else:
            # Multi-page format
            # For ScanImage with Z-stacks, override frame count
            if self._is_scanimage and self._z_stack_info:
                self.frame_count = self._z_stack_info["total_frames_flattened"]
            else:
                self.frame_count = len(pages)

            self.height = first_page.imagelength
            self.width = first_page.imagewidth

            # Determine channels from samples per pixel
            if hasattr(first_page, "samplesperpixel"):
                self.n_channels = first_page.samplesperpixel
            else:
                self.n_channels = (
                    1 if len(first_page.shape) == 2 else first_page.shape[2]
                )

        # Apply deinterleaving
        if self.deinterleave > 1:
            self.n_channels = self.deinterleave
            self.frame_count = self.frame_count // self.deinterleave

        # Get data type
        self.dtype = first_page.dtype

    def _read_raw_frames(self, frame_indices: Union[slice, List[int]]) -> np.ndarray:
        """
        Read raw frames from TIFF file.

        Args:
            frame_indices: 0-based indices (slice or list)

        Returns:
            Array with shape (T, H, W, C)
        """
        # Convert slice to list
        if isinstance(frame_indices, slice):
            start, stop, step = frame_indices.indices(self.frame_count)
            frame_indices = list(range(start, stop, step))

        n_frames = len(frame_indices)
        if n_frames == 0:
            return np.empty(
                (0, self.height, self.width, self.n_channels), dtype=self.dtype
            )

        # Allocate output
        output = np.zeros(
            (n_frames, self.height, self.width, self.n_channels), dtype=self.dtype
        )

        # if self._sample_mode:
        #    # Single page with multiple samples - read strips
        #    self._read_sample_mode(frame_indices, output)
        # elif self._tiff_series is not None and not self.deinterleave > 1:
        #    # Use series for efficient reading
        #    self._read_series_mode(frame_indices, output)
        # else:
        #    # Page-based reading (with possible deinterleaving)
        self._read_page_mode(frame_indices, output)

        return output

    def _read_sample_mode(self, frame_indices: List[int], output: np.ndarray):
        """Read frames in sample mode (single page, multiple samples)."""
        page = self._tiff_file.pages[0]

        # Read the entire page once
        data = page.asarray()

        # Extract requested samples (treating samples as frames)
        for i, frame_idx in enumerate(frame_indices):
            if len(data.shape) == 3:
                # Data is (H, W, Samples)
                output[i, :, :, 0] = data[:, :, frame_idx]
            else:
                # Data is (H, W)
                output[i, :, :, 0] = data

    def _read_series_mode(self, frame_indices: List[int], output: np.ndarray):
        """Read frames using tifffile series (efficient for large stacks)."""
        # Get full data array (may be memory-mapped)
        if self.use_memmap:
            data = self._tiff_series.asarray(out="memmap")
        else:
            data = self._tiff_series.asarray()

        # Parse axes to find dimensions
        axes = self._tiff_series.axes

        # Build indexing based on axes
        for i, frame_idx in enumerate(frame_indices):
            if "T" in axes:
                t_idx = axes.index("T")
                if "C" in axes:
                    # Handle different axis orders
                    if axes == "TCYX":
                        output[i] = np.transpose(data[frame_idx], (1, 2, 0))
                    elif axes == "TYX":
                        output[i, :, :, 0] = data[frame_idx]
                    elif axes == "TYXC":
                        output[i] = data[frame_idx]
                    else:
                        # Generic handling
                        frame_data = np.take(data, frame_idx, axis=t_idx)
                        output[i] = self._reshape_to_output(frame_data, axes)
                else:
                    # No channel dimension
                    output[i, :, :, 0] = np.take(data, frame_idx, axis=t_idx)
            else:
                # No time dimension, treat pages as frames
                output[i] = self._page_to_frame(data, frame_idx)

    def _read_page_mode(self, frame_indices: List[int], output: np.ndarray):
        """Read frames page by page (with deinterleaving support)."""
        for i, frame_idx in enumerate(frame_indices):
            if self.deinterleave > 1:
                # Read multiple pages for deinterleaved channels
                base_page_idx = frame_idx * self.deinterleave
                for ch in range(self.deinterleave):
                    page_idx = base_page_idx + ch
                    if page_idx < len(self._tiff_file.pages):
                        page_data = self._tiff_file.pages[page_idx].asarray()
                        if page_data.ndim == 2:
                            output[i, :, :, ch] = page_data
                        else:
                            output[i, :, :, ch] = page_data[:, :, 0]
            else:
                # Standard page reading
                page = self._tiff_file.pages[frame_idx]
                page_data = page.asarray()

                if page_data.ndim == 2:
                    output[i, :, :, 0] = page_data
                elif page_data.ndim == 3:
                    # Multi-channel page
                    n_ch = min(page_data.shape[2], self.n_channels)
                    output[i, :, :, :n_ch] = page_data[:, :, :n_ch]

    def _reshape_to_output(self, data: np.ndarray, axes: str) -> np.ndarray:
        """Reshape data from arbitrary axes order to (H, W, C)."""
        # Find indices
        y_idx = axes.index("Y") if "Y" in axes else 0
        x_idx = axes.index("X") if "X" in axes else 1
        c_idx = (
            axes.index("C") if "C" in axes else axes.index("S") if "S" in axes else None
        )

        if c_idx is not None:
            # Transpose to (Y, X, C)
            perm = [y_idx, x_idx, c_idx]
            remaining = [i for i in range(len(axes)) if i not in perm]
            perm.extend(remaining)
            result = np.transpose(data, perm)
            return result[:, :, : self.n_channels]
        else:
            # No channel dimension
            if data.ndim == 2:
                return data[:, :, np.newaxis]
            else:
                perm = [y_idx, x_idx]
                remaining = [i for i in range(len(axes)) if i not in perm]
                perm.extend(remaining)
                result = np.transpose(data, perm)
                return result[:, :, np.newaxis]

    def _page_to_frame(self, data: np.ndarray, idx: int) -> np.ndarray:
        """Convert page data to frame format (H, W, C)."""
        if data.ndim == 2:
            return data[:, :, np.newaxis]
        elif data.ndim == 3:
            return data[idx] if data.shape[0] > idx else data
        else:
            return data

    def close(self):
        """Close TIFF file."""
        if self._tiff_file:
            self._tiff_file.close()
            self._tiff_file = None
            self._tiff_series = None

    def get_metadata(self) -> dict:
        """Get comprehensive metadata from TIFF file."""
        self._ensure_initialized()

        metadata = {
            "file_name": os.path.basename(self.file_path),
            "frame_count": self.frame_count,
            "shape": self.shape,
            "unbinned_shape": self.unbinned_shape,
            "dtype": str(self.dtype),
            "sample_mode": self._sample_mode,
            "deinterleave": self.deinterleave,
            "is_scanimage": self._is_scanimage,
        }

        # Add ScanImage Z-stack information if present
        if self._z_stack_info:
            metadata["z_stack_info"] = self._z_stack_info
            metadata["scanimage_version"] = self._scanimage_metadata.get("version")
            metadata["z_step_microns"] = self._z_stack_info.get("z_step")

        # Add TIFF-specific metadata if available
        if self._tiff_file:
            first_page = self._tiff_file.pages[0]
            if hasattr(first_page, "tags"):
                # Extract common tags
                tags = first_page.tags
                if "ImageDescription" in tags:
                    metadata["description"] = str(tags["ImageDescription"].value)[
                        :500
                    ]  # Limit length
                if "Software" in tags:
                    metadata["software"] = str(tags["Software"].value)
                if "DateTime" in tags:
                    metadata["datetime"] = str(tags["DateTime"].value)

        return metadata


class TIFFFileWriter(VideoWriter):
    """
    TIFF stack file writer with multi-page and compression support.

    Features:
    - Multi-page TIFF writing
    - Suite2p format support (interleaved single-channel pages)
    - Various compression algorithms
    - ImageJ metadata compatibility
    - BigTIFF support for files >4GB (default: always enabled like MATLAB)
    """

    def __init__(self, file_path: str, **kwargs):
        """
        Initialize TIFF writer.

        Args:
            file_path: Output file path
            format: 'default' or 'suite2p' (changes to interleaved single-channel pages)
            compression: Compression type ('none', 'lzw', 'zlib', 'jpeg')
            compression_level: Compression level for zlib (0-9)
            bigtiff: Use BigTIFF format (default: True, matching MATLAB 'w8')
            imagej: Write ImageJ-compatible metadata
            metadata: Additional metadata dict to include
        """
        if not TIFF_SUPPORTED:
            raise ImportError("tifffile library required for TIFF support")

        # Initialize parent classes
        VideoWriter.__init__(self)

        self.file_path = file_path
        self._frame_count = 0

        # Options
        self.format = kwargs.get("format", "default")
        self.compression = kwargs.get("compression", "none")
        self.compression_level = kwargs.get("compression_level", 6)
        self.bigtiff = kwargs.get("bigtiff", True)  # Default to True like MATLAB 'w8'
        self.imagej = kwargs.get("imagej", False)
        self.metadata = kwargs.get("metadata", {})

        # Compression mapping
        self._compression_map = {
            "none": None,
            "lzw": "lzw",
            "zlib": "zlib",
            "deflate": "zlib",
            "jpeg": "jpeg",
        }

        # Track axes for current write
        self._current_axes = None

    def write_frames(self, frames: np.ndarray):
        """
        Write frames to TIFF file.

        Args:
            frames: Array with shape (T, H, W, C) or (T, H, W) or (H, W)
        """
        # Normalize input to 4D (T, H, W, C)
        if frames.ndim == 2:  # Single frame, single channel
            frames = frames[np.newaxis, :, :, np.newaxis]
        elif frames.ndim == 3:
            if len(frames) == 1 or (
                hasattr(self, "height")
                and frames.shape[0] == self.height
                and frames.shape[1] == self.width
            ):
                # Single frame, multiple channels (H, W, C) or (1, H, W)
                if len(frames) == 1:
                    frames = frames[:, :, :, np.newaxis]
                else:
                    frames = frames[np.newaxis, :, :, :]
            else:
                # Multiple frames, single channel (T, H, W)
                frames = frames[:, :, :, np.newaxis]
        elif frames.ndim != 4:
            raise ValueError(f"Expected 2D, 3D or 4D input, got {frames.ndim}D")

        T, H, W, C = frames.shape

        # Apply format transformations BEFORE initialization
        if self.format == "suite2p" and C > 1:
            # Suite2p: interleave channels as single-channel pages
            frames = self._format_suite2p(frames)
            T, H, W, C = frames.shape

        # Now squeeze single channel dimension for proper axes
        if C == 1:
            frames = frames[:, :, :, 0]  # Remove channel dimension
            self._current_axes = "TYX"
        else:
            self._current_axes = "TCYX"
            frames = np.moveaxis(frames, -1, -3)

        # Initialize on first write (after format transforms)
        if not self.initialized:
            self.height = H
            self.width = W
            self.n_channels = 1 if self._current_axes == "TYX" else C
            self.dtype = frames.dtype
            self.initialized = True
            self._create_file()

        # Validate shape consistency
        if H != self.height or W != self.width:
            raise ValueError(
                f"Frame size mismatch. Expected ({self.height}, {self.width}), "
                f"got ({H}, {W})"
            )

        expected_c = 1 if self._current_axes == "TYX" else C
        actual_c = 1 if frames.ndim == 3 else frames.shape[1]
        if actual_c != expected_c:
            raise ValueError(
                f"Channel count mismatch. Expected {expected_c}, got {actual_c}"
            )

        # Write frames

        self._write_frames_to_file(frames)

        # Update frame count based on original T (before suite2p transform)
        if self.format == "suite2p" and self.n_channels == 1:
            # For suite2p, T is already multiplied by original channels
            self._frame_count += len(frames)
        else:
            self._frame_count += T

    def _create_file(self):
        """Create and initialize TIFF file."""
        # Remove existing file
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        # Set up file parameters
        self._setup_write_params()

    def _setup_write_params(self):
        """Set up parameters for writing."""
        # Set up compression
        compression = self._compression_map.get(self.compression.lower(), None)

        if compression == "zlib":
            compression = (compression, self.compression_level)

        # Create metadata with current axes
        metadata = self.metadata.copy()
        if hasattr(self, "_current_axes"):
            metadata.update(
                {
                    # 'axes': self._current_axes,
                    "Software": "pyflowreg"
                }
            )

        self._file_kwargs = {
            "bigtiff": self.bigtiff,
            "compression": compression,
            "metadata": metadata
            if hasattr(self, "_current_axes")
            else {"Software": "pyflowreg"},
            "imagej": self.imagej,
        }

        self._frame_kwargs = {
            "compression": compression,
            "metadata": metadata
            if hasattr(self, "_current_axes")
            else None,  # ✅ Fallback
        }

    def _format_suite2p(self, frames: np.ndarray) -> np.ndarray:
        """
        Convert frames to Suite2p format (interleaved single-channel pages).

        Args:
            frames: (T, H, W, C) array

        Returns:
            (T*C, H, W, 1) array with interleaved channels
        """
        T, H, W, C = frames.shape
        if C == 1:
            return frames

        # Interleave channels as separate pages
        # Order: frame0_ch0, frame0_ch1, ..., frame1_ch0, frame1_ch1, ...
        formatted = np.zeros((T * C, H, W, 1), dtype=frames.dtype)
        for t in range(T):
            for c in range(C):
                formatted[t * C + c, :, :, 0] = frames[t, :, :, c]

        return formatted

    def _write_frames_to_file(self, frames: np.ndarray):
        """
        Write frame data to TIFF file.

        Args:
            frames: Array with shape (T, H, W) or (T, H, W, C)
        """
        # Write using tifffile
        # if self._frame_count == 0:
        # First write - create new file with all kwargs

        # tifffile.imwrite(
        ##     self.file_path,
        ##      frames,
        ##       planarconfig='contig',
        #       contiguous=True,
        #       **self._file_kwargs
        #   )
        ##else:
        # Append to existing file
        append = self._frame_count > 0
        with tifffile.TiffWriter(
            self.file_path, append=append, bigtiff=self.bigtiff, imagej=self.imagej
        ) as tif:
            # Write each frame individually for append mode
            if frames.ndim == 3:  # (T, H, W)
                for frame in frames:
                    tif.write(frame[None, ...], **self._frame_kwargs)
            else:  # (T, C, H, W)
                for frame in frames:
                    frame = np.moveaxis(frame, 0, -1)  # Move C
                    tif.write(
                        frame,
                        planarconfig="contig",
                        contiguous=True,
                        **self._frame_kwargs,
                    )

    def close(self):
        """Close the TIFF file."""
        if self._frame_count > 0:
            print(f"TIFF file written: {self.file_path}")
            print(f"  Frames: {self._frame_count}")
            print(f"  Dimensions: {self.height}x{self.width}")
            print(f"  Channels: {self.n_channels}")
            print(f"  Format: {self.format}")
            print(f"  Compression: {self.compression}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Test functions
def test_basic_functionality():
    """Test basic TIFF reading and writing."""
    import numpy as np

    print("Testing basic TIFF functionality...")

    # Create test data
    test_data = np.random.randint(0, 65535, (10, 256, 256, 2), dtype=np.uint16)

    # Test multi-channel writing
    print("\nTesting multi-channel TIFF...")
    with TIFFFileWriter("test_multichannel.tif", compression="lzw") as writer:
        writer.write_frames(test_data[:5])
        writer.write_frames(test_data[5:])

    # Test single-channel writing
    print("\nTesting single-channel TIFF...")
    single_channel = test_data[:, :, :, 0:1]
    with TIFFFileWriter("test_singlechannel.tif", compression="lzw") as writer:
        writer.write_frames(single_channel)

    # Test Suite2p format
    print("\nTesting Suite2p format...")
    with TIFFFileWriter("test_suite2p.tif", format="suite2p") as writer:
        writer.write_frames(test_data[:5])

    # Test reading back
    print("\nTesting reading...")
    reader = TIFFFileReader("test_multichannel.tif")
    print(f"  Shape: {reader.shape}")
    print(f"  Dtype: {reader.dtype}")

    frame = reader[0]
    print(f"  Single frame shape: {frame.shape}")

    frames = reader[0:3]
    print(f"  Multiple frames shape: {frames.shape}")

    reader.close()

    print("\nBasic tests completed successfully!")


def test_mdf_conversion():
    """Test MDF to TIFF conversion with proper binning."""
    import numpy as np
    from pathlib import Path

    try:
        from pyflowreg.util.io.mdf import MDFFileReader
    except ImportError:
        print("MDF reader not available, skipping MDF conversion test")
        return

    import cv2

    # Input and output paths
    filename = r"D:\2025_OIST\Shinobu\RFPonly\190403_001.MDF"
    out_path = Path(filename).with_suffix(".tif")

    print("Converting MDF to TIFF with proper binning...")
    print(f"  Input: {filename}")
    print(f"  Output: {out_path}")

    # Method 1: Write unbinned, then compare binned reads
    print("\nMethod 1: Write unbinned TIFF, bin during read")

    mdf_unbinned = MDFFileReader(filename, buffer_size=500, bin_size=1)

    # Write subset of unbinned frames
    start_frame = 5 * 8200
    end_frame = 5 * 8300

    with TIFFFileWriter(str(out_path), compression="lzw", bigtiff=True) as writer:
        print(f"Writing frames {start_frame} to {end_frame}...")

        for i in range(start_frame, end_frame):
            if (i - start_frame) % 20 == 0:
                print(f"  Frame {i - start_frame + 1}/{end_frame - start_frame}")

            frame = mdf_unbinned[i]
            if frame.ndim == 2:
                frame = frame[np.newaxis, :, :]
            elif frame.ndim == 3 and frame.shape[0] != 1:
                frame = frame[np.newaxis, :, :, :]
            writer.write_frames(frame)

    mdf_unbinned.close()

    # Now compare binned versions
    print("\nComparing binned versions...")

    # Read with binning
    tiff_reader = TIFFFileReader(str(out_path), buffer_size=500, bin_size=5)
    mdf_binned = MDFFileReader(filename, buffer_size=500, bin_size=5)

    # Read corresponding frames
    tiff_frames = tiff_reader[0:20]  # First 20 binned frames from TIFF
    mdf_frames = mdf_binned[8200:8220]  # Corresponding binned frames from MDF

    # Check equality
    if np.allclose(tiff_frames, mdf_frames, rtol=1e-5, atol=1):
        print("✓ Frames match within tolerance (expected due to rounding)")
    else:
        max_diff = np.abs(tiff_frames.astype(float) - mdf_frames.astype(float)).max()
        print(f"⚠ Max difference: {max_diff:.3f}")

    # Visual comparison
    print("\nShowing visual comparison (press ESC to exit)...")
    for i in range(min(5, tiff_frames.shape[0])):
        tiff_norm = cv2.normalize(
            tiff_frames[i, :, :, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        mdf_norm = cv2.normalize(
            mdf_frames[i, :, :, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        comparison = np.hstack([tiff_norm, mdf_norm])
        cv2.putText(comparison, "TIFF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.putText(
            comparison,
            "MDF",
            (tiff_norm.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
            2,
        )

        cv2.imshow("TIFF vs MDF (binned)", comparison)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Method 2: Pre-bin in float, then write
    print("\nMethod 2: Pre-bin in float domain, then write")

    out_path_binned = (
        Path(filename).with_stem(Path(filename).stem + "_binned").with_suffix(".tif")
    )

    mdf_unbinned = MDFFileReader(filename, buffer_size=500, bin_size=1)

    with TIFFFileWriter(
        str(out_path_binned), compression="lzw", bigtiff=True
    ) as writer:
        print("Writing pre-binned frames...")

        # Process in chunks of bin_size
        for chunk_start in range(start_frame, end_frame, 5):
            chunk_end = min(chunk_start + 5, end_frame)

            # Read chunk and average in float
            chunk_frames = []
            for i in range(chunk_start, chunk_end):
                frame = mdf_unbinned[i]
                chunk_frames.append(frame.astype(np.float32))

            if chunk_frames:
                # Average and convert back to original dtype
                avg_frame = np.mean(chunk_frames, axis=0)
                avg_frame = np.round(avg_frame).astype(mdf_unbinned.dtype)

                if avg_frame.ndim == 2:
                    avg_frame = avg_frame[np.newaxis, :, :]
                elif avg_frame.ndim == 3 and avg_frame.shape[0] != 1:
                    avg_frame = avg_frame[np.newaxis, :, :, :]

                writer.write_frames(avg_frame)

    print(f"\nPre-binned TIFF written: {out_path_binned}")

    # Cleanup
    tiff_reader.close()
    mdf_binned.close()
    mdf_unbinned.close()

    print("\nConversion tests completed!")


def main2():
    import numpy as np
    from pathlib import Path
    from mdf import MDFFileReader
    import cv2

    filename = r"D:\2025_OIST\Shinobu\RFPonly\190403_001.MDF"
    out_path = Path(filename + ".tiff")

    mdf = MDFFileReader(filename, buffer_size=500, bin_size=1)

    with TIFFFileWriter(str(out_path)) as w:
        # for i in range(5 * 8200, 5 * 9200):
        for i in range(5 * 8200, 5 * 8221):
            frame = mdf[i]
            w.write_frames(frame[np.newaxis])

    h5 = TIFFFileReader(str(out_path), buffer_size=500, bin_size=5)
    h5_b5 = h5[0:20]
    h5.close()
    mdf.close()

    mdf2 = MDFFileReader(filename, buffer_size=500, bin_size=5)
    mdf_b5 = mdf2[8200 : 8200 + 20]
    mdf2.close()

    counter = 0
    while True:
        frame = np.concatenate([h5_b5[counter], mdf_b5[counter]], axis=0)
        counter = (counter + 1) % h5_b5.shape[0]
        cv2.imshow(
            "Frame",
            cv2.normalize(
                frame[..., 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            ),
        )
        key = cv2.waitKey(1)
        if key == 27:
            break

    if not np.array_equal(h5_b5, mdf_b5):
        d = h5_b5.astype(np.int64) - mdf_b5.astype(np.int64)
        print(int(np.abs(d).max()))
        print("Frames are not equal!")
    else:
        print(f"OK {out_path}")


def main3():
    input_file = "D:\\2024_OIST\\flow-registration\\2024\\MotionCorrection_2024\\FearConditioning_L6_somas_z1\\M231221_2_240207_002_001.TIF"
    import cv2

    tif_reader = TIFFFileReader(input_file, bin_size=20)
    frames = tif_reader[0:500]
    counter = 0
    while True:
        cv2.imshow(
            "Frame",
            cv2.normalize(
                frames[counter], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            ),
        )
        counter = (counter + 1) % frames.shape[0]
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == "__main__":
    main3()
