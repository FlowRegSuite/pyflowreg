# File Formats

PyFlowReg provides a modular I/O system supporting multiple file formats through factory functions and a common VideoReader/VideoWriter interface.

## Supported Formats

### HDF5 (.h5, .hdf5, .hdf)

**Recommended for large datasets**

```python
from pyflowreg.motion_correction import compensate_recording, OFOptions

options = OFOptions(
    input_file="video.h5",
    output_path="results/",
    output_format="HDF5"  # Creates separate 3D datasets per channel
)
compensate_recording(options)
```

**Features:**
- **MATLAB compatibility**: Stores channels as separate 3D datasets (`ch1`, `ch2`, etc.)
- **Compression**: Optional gzip or lzf compression
- **Chunked I/O**: Efficient reading/writing of large files
- **Metadata**: Stores dimension ordering and shape info as HDF5 attributes
- **Multi-channel**: Each channel as separate dataset for MATLAB interop

**Storage format:**
- Dimension ordering: `(H, W, T)` by default (MATLAB convention)
- Dataset names: `ch1`, `ch2`, ..., `chN`
- Attributes: `frame_count`, `height`, `width`, `n_channels`, `dimension_ordering`

### TIFF (.tif, .tiff)

**Standard microscopy format**

```python
options = OFOptions(
    input_file="video.tif",
    output_format="TIFF"
)
```

**Features:**
- **Multi-page stacks**: Standard multi-page TIFF
- **ScanImage support**: Auto-detects and parses ScanImage metadata
- **Channel handling**: Auto-deinterleaving for ScanImage multi-channel
- **Memory mapping**: Optional memory-mapped reading for large files
- **Z-stack awareness**: Detects and handles volumetric data

**Supported variants:**
- Standard TIFF stacks
- ImageJ/Fiji hyperstacks
- ScanImage TIFF (with metadata parsing)
- Multi-sample TIFF (channels as samples per pixel)

### MATLAB MAT (.mat)

**Bidirectional MATLAB compatibility**

```python
options = OFOptions(
    input_file="video.mat",
    output_format="MAT"  # Default output format
)
```

**Features:**
- **Full MATLAB parity**: Maintains algorithm compatibility with Flow-Registration
- **v7.3 format**: HDF5-based MAT files for large datasets
- **Variable name**: Uses `"mov"` by default or first variable found

**Default behavior:**
- Output format defaults to `"MAT"` for MATLAB compatibility
- Automatically handles v7 and v7.3 format differences

### MDF (Sutter MesaScope)

**Windows-only, requires pywin32**

```python
options = OFOptions(
    input_file="recording.mdf",
    output_format="HDF5"  # Convert MDF to HDF5
)
```

**Features:**
- **Native Sutter format**: Reads MesaScope .MDF files directly
- **Metadata extraction**: Preserves acquisition parameters
- **Windows only**: Requires `pywin32` package
- **Read-only**: No MDF writer (convert to HDF5 or TIFF for output)

## Special Output Formats

### Multi-File Formats

Split large outputs across multiple files:

```python
options = OFOptions(
    input_file="large_video.h5",
    output_format="MULTIFILE_HDF5"  # or MULTIFILE_TIFF, MULTIFILE_MAT
)
```

Creates separate files per channel: `compensated_ch1.h5`, `compensated_ch2.h5`, etc.

### CaImAn Compatibility

```python
options = OFOptions(
    output_format="CAIMAN_HDF5"  # Multi-file HDF5 with /mov dataset
)
```

Creates HDF5 files compatible with CaImAn's expected format.

### Suite2p Compatibility

```python
options = OFOptions(
    output_format="SUITE2P_TIFF"
)
```

## Factory Functions

### Creating Readers

```python
from pyflowreg.util.io import get_video_file_reader

# Automatic format detection from extension
reader = get_video_file_reader("video.h5", buffer_size=500, bin_size=1)

# Numpy array input
reader = get_video_file_reader(video_array)

# Multi-channel from separate files
reader = get_video_file_reader(["ch1.tif", "ch2.tif"])
```

### Creating Writers

```python
from pyflowreg.util.io import get_video_file_writer

# Create writer by format
writer = get_video_file_writer("output.h5", "HDF5")
writer.write_frames(frames)  # frames: (T, H, W, C)
writer.close()
```

## Configuration Options

### Buffer Size and Binning

```python
options = OFOptions(
    buffer_size=400,  # Number of frames per batch (default: 400)
    bin_size=1  # Temporal binning factor (default: 1)
)
```

- `buffer_size`: Controls memory usage and I/O efficiency (larger = faster but more memory)
- `bin_size`: Temporal binning applied during reading (bin_size=2 averages every 2 frames)

### Output Data Type

```python
options = OFOptions(
    output_typename="double"  # Default: MATLAB-compatible double precision
)
```

Supported types: `"double"` (float64), `"single"` (float32), `"uint16"`, `"uint8"`

### Saving Displacement Fields

```python
options = OFOptions(
    save_w=True,  # Save displacement fields (default: False)
    output_format="HDF5"
)
```

Creates additional file: `<basename>_w.h5` with displacement fields shape `(T, H, W, 2)`

### File Naming

```python
options = OFOptions(
    naming_convention="default",  # or "batch"
    output_file_name="my_output.h5"  # Optional custom name
)
```

- `"default"`: `compensated.<ext>`
- `"batch"`: `<input_basename>_compensated.<ext>`
- Custom: Override with `output_file_name`

## Reading Data

### Array-Like Indexing

All readers support numpy-style indexing:

```python
from pyflowreg.util.io import get_video_file_reader

reader = get_video_file_reader("video.h5")

# Single frame: returns (H, W, C)
frame = reader[0]

# Slice: returns (T, H, W, C)
frames = reader[10:20]

# List indexing: returns (T, H, W, C)
frames = reader[[0, 10, 20, 30]]

# Spatial subset
frames = reader[0:10, :, 100:200, 100:200, :]
```

### Batch Iteration

```python
reader = get_video_file_reader("video.h5", buffer_size=100)

for batch in reader:
    # batch shape: (100, H, W, C) or fewer for last batch
    process(batch)
```

### Multi-Channel from Separate Files

```python
from pyflowreg.util.io.multifile_wrappers import MULTICHANNELFileReader

# List of single-channel files
reader = MULTICHANNELFileReader(["ch1.tif", "ch2.tif"])
frames = reader[:]  # Shape: (T, H, W, 2)
```

## Writing Data

### Batch Writing

```python
from pyflowreg.util.io import get_video_file_writer

writer = get_video_file_writer("output.h5", "HDF5")

# Write in batches
for batch in video_batches:
    writer.write_frames(batch)  # batch: (T, H, W, C)

writer.close()
```

### Context Manager

```python
with get_video_file_writer("output.h5", "HDF5") as writer:
    writer.write_frames(frames)
# Automatically closed
```

## Format Conversion

### Simple Conversion

```python
from pyflowreg.util.io import get_video_file_reader, get_video_file_writer

# Read TIFF
reader = get_video_file_reader("input.tif")

# Write as HDF5
with get_video_file_writer("output.h5", "HDF5") as writer:
    for batch in reader:
        writer.write_frames(batch)

reader.close()
```

### Batch Conversion

```python
from pathlib import Path

input_dir = Path("tiff_files/")
output_dir = Path("hdf5_files/")
output_dir.mkdir(exist_ok=True)

for tiff_file in input_dir.glob("*.tif"):
    reader = get_video_file_reader(str(tiff_file))
    output_file = output_dir / f"{tiff_file.stem}.h5"

    with get_video_file_writer(str(output_file), "HDF5") as writer:
        for batch in reader:
            writer.write_frames(batch)

    reader.close()
    print(f"Converted: {tiff_file.name}")
```

## Performance Tips

### HDF5 Optimization

```python
# Use compression for storage savings
writer = get_video_file_writer("output.h5", "HDF5",
                                compression="gzip",
                                compression_opts=4)  # Level 1-9

# Adjust chunk size for access pattern
# Default chunk_size=1 optimizes for temporal access
writer = get_video_file_writer("output.h5", "HDF5", chunk_size=10)
```

### Memory-Mapped TIFF

```python
# Enable memory mapping for large TIFF files
reader = get_video_file_reader("large.tif", use_memmap=True)
```

### Buffer Size Tuning

```python
# Larger buffers = fewer I/O operations but more memory
# Smaller buffers = more I/O but less memory

# For large RAM and fast disk:
options = OFOptions(buffer_size=1000)

# For limited RAM:
options = OFOptions(buffer_size=100)
```
