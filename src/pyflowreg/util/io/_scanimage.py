"""
Utility functions for parsing and handling ScanImage TIFF metadata.

ScanImage stores microscopy metadata in TIFF files, including information about:
- Multi-channel imaging
- Z-stacks (3D volumes)
- Time series
- ROIs (Regions of Interest)
- Acquisition parameters

This module provides tools to extract and interpret this metadata correctly.
"""

import warnings
from typing import Dict, Any, Tuple

try:
    import tifffile

    TIFFFILE_SUPPORTED = True
except ImportError:
    TIFFFILE_SUPPORTED = False


def parse_scanimage_metadata(file_path: str) -> Dict[str, Any]:
    """
    Parse ScanImage metadata from a TIFF file.

    Args:
        file_path: Path to the ScanImage TIFF file

    Returns:
        Dictionary containing parsed metadata with keys:
        - is_scanimage: Whether this is a ScanImage file
        - version: ScanImage version
        - channels: Number of channels
        - volumes: Number of volumes/stacks
        - slices_per_volume: Number of Z slices per volume
        - frames_per_slice: Number of frames at each Z position
        - total_frames: Total number of 2D frames (volumes * slices_per_volume)
        - z_step: Z step size in microns (if available)
        - frame_rate: Acquisition frame rate (if available)
        - roi_data: ROI information (if available)
        - raw_metadata: Complete raw metadata dictionary
    """
    if not TIFFFILE_SUPPORTED:
        raise ImportError("tifffile library required for ScanImage support")

    metadata = {
        "is_scanimage": False,
        "version": None,
        "channels": 1,
        "volumes": 1,
        "slices_per_volume": 1,
        "frames_per_slice": 1,
        "total_frames": None,
        "z_step": None,
        "frame_rate": None,
        "roi_data": None,
        "raw_metadata": {},
    }

    try:
        with tifffile.TiffFile(file_path) as tif:
            # Check if it's a ScanImage file
            if not hasattr(tif, "is_scanimage") or not tif.is_scanimage:
                return metadata

            metadata["is_scanimage"] = True

            # Get the raw metadata
            if hasattr(tif, "scanimage_metadata"):
                si_metadata = tif.scanimage_metadata
                metadata["raw_metadata"] = si_metadata

                # Parse version
                if "SI" in si_metadata:
                    si = si_metadata["SI"]

                    # Version info
                    if "VERSION_MAJOR" in si:
                        major = si.get("VERSION_MAJOR", "")
                        minor = si.get("VERSION_MINOR", "")
                        metadata["version"] = f"{major}.{minor}"

                    # Channel information
                    if "hChannels" in si:
                        channels = si["hChannels"]
                        if "channelSave" in channels:
                            # channelSave is a list of active channel indices
                            channel_list = channels["channelSave"]
                            if isinstance(channel_list, list):
                                metadata["channels"] = len(channel_list)
                            else:
                                metadata["channels"] = 1

                    # Stack/Volume information
                    if "hStackManager" in si:
                        stack = si["hStackManager"]

                        # Number of slices (Z planes) per volume
                        if "numSlices" in stack:
                            metadata["slices_per_volume"] = int(stack["numSlices"])

                        # Frames per slice (for averaging at each Z position)
                        if "framesPerSlice" in stack:
                            metadata["frames_per_slice"] = int(stack["framesPerSlice"])

                        # Z step size
                        if "stackZStepSize" in stack:
                            metadata["z_step"] = float(stack["stackZStepSize"])

                    # Fast Z information (alternative Z scanning mode)
                    if "hFastZ" in si:
                        fastz = si["hFastZ"]
                        if "enable" in fastz and fastz["enable"]:
                            if "numFramesPerVolume" in fastz:
                                metadata["slices_per_volume"] = int(
                                    fastz["numFramesPerVolume"]
                                )
                            if "numVolumes" in fastz:
                                metadata["volumes"] = int(fastz["numVolumes"])

                    # Frame rate information
                    if "hRoiManager" in si:
                        roi = si["hRoiManager"]
                        if "scanFrameRate" in roi:
                            metadata["frame_rate"] = float(roi["scanFrameRate"])

                # Alternative metadata location for older ScanImage versions
                elif isinstance(si_metadata, dict):
                    # Try to parse frame-varying metadata
                    if "frameNumbers" in si_metadata:
                        # This is frame-specific metadata
                        pass

                    # Look for Software tag
                    if "Software" in si_metadata:
                        software = si_metadata["Software"]
                        if "ScanImage" in str(software):
                            # Extract version from software string
                            import re

                            match = re.search(r"ScanImage\s+([\d.]+)", str(software))
                            if match:
                                metadata["version"] = match.group(1)

            # Try alternative metadata extraction using pages
            if not metadata["version"] and len(tif.pages) > 0:
                first_page = tif.pages[0]

                # Check ImageDescription tag
                if (
                    hasattr(first_page, "tags")
                    and "ImageDescription" in first_page.tags
                ):
                    desc = first_page.tags["ImageDescription"].value

                    # Try to parse as JSON (newer ScanImage versions)
                    try:
                        if isinstance(desc, bytes):
                            desc = desc.decode("utf-8", errors="ignore")

                        # Look for SI structure in the description
                        if "SI." in desc or "SI =" in desc:
                            metadata["is_scanimage"] = True
                            # Extract key parameters using regex
                            _extract_from_description(desc, metadata)
                    except Exception as e:
                        warnings.warn(f"Warning: Failed to parse ImageDescription: {e}")

                # Check Software tag
                if hasattr(first_page, "tags") and "Software" in first_page.tags:
                    software = str(first_page.tags["Software"].value)
                    if "ScanImage" in software:
                        metadata["is_scanimage"] = True
                        import re

                        match = re.search(r"ScanImage\s+([\d.]+)", software)
                        if match:
                            metadata["version"] = match.group(1)

            # Calculate total frames
            # Total frames = volumes * slices_per_volume * frames_per_slice
            # But for motion correction, we want volumes * slices_per_volume
            # (treating each Z plane as a separate frame)
            if metadata["is_scanimage"]:
                # Get actual frame count from TIFF
                total_pages = len(tif.pages)

                # Calculate expected frames based on metadata
                frames_from_metadata = (
                    metadata["volumes"]
                    * metadata["slices_per_volume"]
                    * metadata["frames_per_slice"]
                )

                # If we have multi-channel, divide by channels
                if metadata["channels"] > 1:
                    # Check if channels are interleaved or separate pages
                    if total_pages == frames_from_metadata * metadata["channels"]:
                        # Channels are separate pages (Suite2p style)
                        metadata["channel_mode"] = "interleaved"
                    else:
                        # Channels are in same page
                        metadata["channel_mode"] = "packed"

                # Set total frames (each Z plane counts as a frame)
                metadata["total_frames"] = (
                    metadata["volumes"] * metadata["slices_per_volume"]
                )

                # Store the actual page count for validation
                metadata["tiff_page_count"] = total_pages

    except Exception as e:
        warnings.warn(f"Error parsing ScanImage metadata: {e}")

    return metadata


def _extract_from_description(description: str, metadata: Dict[str, Any]):
    """
    Extract metadata from ImageDescription string using regex patterns.

    This handles older ScanImage formats where metadata is stored as
    MATLAB-evaluable strings rather than JSON.
    """
    import re

    # Channel information patterns
    patterns = {
        "channels": [
            r"SI\.hChannels\.channelSave\s*=\s*\[([\d\s,]+)\]",
            r"SI\.hChannels\.channelsActive\s*=\s*(\d+)",
        ],
        "slices": [
            r"SI\.hStackManager\.numSlices\s*=\s*(\d+)",
            r"SI\.hFastZ\.numFramesPerVolume\s*=\s*(\d+)",
        ],
        "volumes": [
            r"SI\.hFastZ\.numVolumes\s*=\s*(\d+)",
            r"SI\.hStackManager\.numVolumes\s*=\s*(\d+)",
        ],
        "frames_per_slice": [
            r"SI\.hStackManager\.framesPerSlice\s*=\s*(\d+)",
        ],
        "z_step": [
            r"SI\.hStackManager\.stackZStepSize\s*=\s*([\d.]+)",
            r"SI\.hFastZ\.positionAbsolute\s*=\s*\[([\d.\s,]+)\]",
        ],
        "frame_rate": [
            r"SI\.hRoiManager\.scanFrameRate\s*=\s*([\d.]+)",
        ],
    }

    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, description)
            if match:
                if key == "channels":
                    # Handle channel list
                    if "[" in pattern:
                        channel_str = match.group(1)
                        channels = [
                            int(x) for x in channel_str.replace(",", " ").split()
                        ]
                        metadata["channels"] = len(channels)
                    else:
                        metadata["channels"] = int(match.group(1))
                elif key == "slices":
                    metadata["slices_per_volume"] = int(match.group(1))
                elif key == "volumes":
                    metadata["volumes"] = int(match.group(1))
                elif key == "frames_per_slice":
                    metadata["frames_per_slice"] = int(match.group(1))
                elif key == "z_step":
                    if "[" in pattern:
                        # Extract Z positions and calculate step
                        z_str = match.group(1)
                        z_positions = [
                            float(x) for x in z_str.replace(",", " ").split()
                        ]
                        if len(z_positions) > 1:
                            metadata["z_step"] = abs(z_positions[1] - z_positions[0])
                    else:
                        metadata["z_step"] = float(match.group(1))
                elif key == "frame_rate":
                    metadata["frame_rate"] = float(match.group(1))
                break


def interpret_scanimage_dimensions(
    tif_shape: Tuple[int, ...], tif_axes: str, si_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Interpret dimensions of a ScanImage TIFF based on shape, axes, and metadata.

    Args:
        tif_shape: Shape tuple from tifffile
        tif_axes: Axes string from tifffile (e.g., 'TZYX', 'TYX', 'ZYX')
        si_metadata: Parsed ScanImage metadata dictionary

    Returns:
        Dictionary with interpreted dimensions:
        - total_frames: Total 2D frames for processing
        - height: Image height
        - width: Image width
        - channels: Number of channels
        - volumes: Number of 3D volumes
        - z_planes: Number of Z planes per volume
        - true_time_frames: Actual time points (volumes)
    """
    result = {
        "total_frames": 1,
        "height": None,
        "width": None,
        "channels": 1,
        "volumes": 1,
        "z_planes": 1,
        "true_time_frames": 1,
        "interpretation": "unknown",
    }

    # Get spatial dimensions
    if "Y" in tif_axes:
        y_idx = tif_axes.index("Y")
        result["height"] = tif_shape[y_idx]
    if "X" in tif_axes:
        x_idx = tif_axes.index("X")
        result["width"] = tif_shape[x_idx]

    # Get channel dimension
    if "C" in tif_axes:
        c_idx = tif_axes.index("C")
        result["channels"] = tif_shape[c_idx]
    elif si_metadata.get("channels"):
        result["channels"] = si_metadata["channels"]

    # Interpret time and Z dimensions based on ScanImage metadata
    if si_metadata.get("is_scanimage"):
        slices = si_metadata.get("slices_per_volume", 1)
        # volumes from metadata available but overridden by tif_shape below
        _volumes_from_metadata = si_metadata.get("volumes", 1)  # noqa: F841

        if "Z" in tif_axes and "T" in tif_axes:
            # Both Z and T present
            z_idx = tif_axes.index("Z")
            t_idx = tif_axes.index("T")

            result["z_planes"] = tif_shape[z_idx]
            result["volumes"] = tif_shape[t_idx]
            result["total_frames"] = result["volumes"] * result["z_planes"]
            result["true_time_frames"] = result["volumes"]
            result["interpretation"] = "time_series_of_volumes"

        elif "Z" in tif_axes:
            # Only Z present - interpret as single volume
            z_idx = tif_axes.index("Z")
            result["z_planes"] = tif_shape[z_idx]
            result["volumes"] = 1
            result["total_frames"] = result["z_planes"]
            result["true_time_frames"] = 1
            result["interpretation"] = "single_volume"

        elif "T" in tif_axes:
            # Only T present
            t_idx = tif_axes.index("T")
            total_t = tif_shape[t_idx]

            if slices > 1:
                # T dimension contains interleaved Z slices
                result["z_planes"] = slices
                result["volumes"] = total_t // slices
                result["total_frames"] = total_t
                result["true_time_frames"] = result["volumes"]
                result["interpretation"] = "interleaved_z_in_t"
            else:
                # Pure time series
                result["volumes"] = total_t
                result["z_planes"] = 1
                result["total_frames"] = total_t
                result["true_time_frames"] = total_t
                result["interpretation"] = "pure_time_series"
    else:
        # Non-ScanImage file - use standard interpretation
        if "T" in tif_axes:
            t_idx = tif_axes.index("T")
            result["total_frames"] = tif_shape[t_idx]
            result["true_time_frames"] = tif_shape[t_idx]
        if "Z" in tif_axes:
            z_idx = tif_axes.index("Z")
            result["z_planes"] = tif_shape[z_idx]
            result["total_frames"] = max(result["total_frames"], tif_shape[z_idx])

    return result


def format_scanimage_metadata_report(metadata: Dict[str, Any]) -> str:
    """
    Format ScanImage metadata into a human-readable report.

    Args:
        metadata: Parsed metadata dictionary

    Returns:
        Formatted string report
    """
    lines = []

    if not metadata.get("is_scanimage"):
        return "Not a ScanImage file"

    lines.append("=== ScanImage File Metadata ===")
    lines.append(f"Version: {metadata.get('version', 'Unknown')}")
    lines.append("")

    lines.append("Acquisition Parameters:")
    lines.append(f"  Channels: {metadata.get('channels', 1)}")
    lines.append(f"  Volumes/Stacks: {metadata.get('volumes', 1)}")
    lines.append(f"  Slices per volume: {metadata.get('slices_per_volume', 1)}")
    lines.append(f"  Frames per slice: {metadata.get('frames_per_slice', 1)}")

    total = metadata.get("total_frames")
    if total:
        lines.append(f"  Total 2D frames: {total}")

    if metadata.get("z_step"):
        lines.append(f"  Z step size: {metadata['z_step']:.2f} µm")

    if metadata.get("frame_rate"):
        lines.append(f"  Frame rate: {metadata['frame_rate']:.2f} Hz")

    if metadata.get("channel_mode"):
        lines.append(f"  Channel mode: {metadata['channel_mode']}")

    if metadata.get("tiff_page_count"):
        lines.append(f"  TIFF pages: {metadata['tiff_page_count']}")

    return "\n".join(lines)


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Parsing ScanImage metadata from: {file_path}")

        metadata = parse_scanimage_metadata(file_path)
        report = format_scanimage_metadata_report(metadata)
        print(report)

        if metadata.get("is_scanimage"):
            print("\nRaw metadata keys:", list(metadata.get("raw_metadata", {}).keys()))
    else:
        print("Usage: python _scanimage.py <path_to_scanimage_tiff>")
