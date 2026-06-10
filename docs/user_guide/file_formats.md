# File Formats

PyFlowReg provides a modular I/O system supporting multiple file formats through factory functions and a common VideoReader/VideoWriter interface. This page is the reference for format support in PyFlowReg; other pages link here instead of restating format details.

Readers are selected by file extension: `.tif`/`.tiff` (TIFF), `.h5`/`.hdf5`/`.hdf` (HDF5), `.mat` (MAT), and `.mdf` (MDF, Windows-only). Files with other extensions are probed as HDF5 before an error is raised. The reader factory also accepts numpy arrays, already-constructed readers, and lists of per-channel files (see [Multi-Channel from Separate Files](#multi-channel-from-separate-files)).

## Supported Formats

### HDF5 (.h5, .hdf5, .hdf)

**Recommended for large datasets**

```{literalinclude} ../snippets/user_guide/file_formats/hdf5_pipeline.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

**Features:**
- **MATLAB-style layout**: The writer stores each channel as a separate 3D dataset (`ch1`, `ch2`, etc.)
- **Compression**: Optional `gzip` or `lzf` compression (writer option, off by default)
- **Chunked, expandable datasets**: Frames are appended batch-by-batch
- **Metadata**: Dimension ordering and shape information stored as HDF5 attributes
- **Dataset discovery**: The reader auto-detects suitable datasets when no `dataset_names` are given (channel-style names like `ch1`, `ch2` first, then common names such as `mov` or `data`, then the largest 3D/4D dataset)

**Storage format (writer defaults):**
- Per-channel 3D datasets stored as `(T, H, W)` (`dimension_ordering=(1, 2, 0)`), configurable via the `dimension_ordering` option
- Dataset names: `ch1`, `ch2`, ..., `chN` (pattern configurable via `dataset_names`)
- File-level attributes: `frame_count`, `height`, `width`, `n_channels`, `dimension_ordering`, `format`, `dataset_names`

The reader interprets 3D datasets as `(T, H, W)` with one dataset per channel, and a 4D dataset as `(T, H, W, C)`.

### TIFF (.tif, .tiff)

**Standard microscopy format**

```python
options = OFOptions(
    input_file="video.tif",
    output_format="TIFF"
)
```

**Reader features:**
- **Multi-page stacks**: Standard multi-page TIFF
- **ScanImage support**: ScanImage files are auto-detected and their metadata parsed to derive the channel count (auto-enabling deinterleaving of interleaved channel pages) and to flatten Z-stacks/volumes into a sequence of 2D frames
- **Manual deinterleaving**: The `deinterleave=N` option treats N consecutive single-channel pages as one N-channel frame (e.g. Suite2p-style files)
- **Memory mapping**: Memory-mapped reading through the tifffile series path (`use_memmap`, default `True`)
- **Series and multi-sample support**: Files exposing a tifffile series (e.g. ImageJ stacks, OME-TIFF) and single-page multi-sample TIFFs (samples per pixel read as frames)

**Writer features:**
- Multi-page output with optional compression (`none`, `lzw`, `zlib`/`deflate`, `jpeg`; default `none`)
- BigTIFF enabled by default (matching MATLAB `'w8'`)
- Optional ImageJ-compatible metadata (`imagej=True`)
- Suite2p-style output (channels as interleaved single-channel pages) via the `SUITE2P_TIFF` output format

### MATLAB MAT (.mat)

**Reading and writing MATLAB MAT-files**

```python
options = OFOptions(
    input_file="video.mat",
    output_format="MAT"  # Default output format
)
```

**Features:**
- **Reader**: Supports traditional MAT-files (v5/v7, read via scipy.io) and v7.3 files (HDF5-based, read via h5py)
- **Dataset discovery**: Channel-style variable names (`ch1`, `ch2`, ...) are found first, then common names (`mov`, `data`, `dataset`), then the largest 3D/4D variable; explicit names can be passed with `dataset_names`
- **Writer**: Stores one 3D array per channel (default names `ch1`, `ch2`, ..., MATLAB `(H, W, T)` ordering) plus a `__pyflowreg_metadata__` entry

**Default behavior:**
- `OFOptions.output_format` defaults to `"MAT"` (matching the MATLAB toolbox)
- The writer uses MAT v5 with compression by default; v7.3 (via hdf5storage) is used when `use_v73=True` or when the data is too large for v5
- The MAT writer buffers all frames in memory and writes the file on `close()`; for long recordings, prefer HDF5 output

### MDF (.mdf)

**Windows-only, requires pywin32**

```python
options = OFOptions(
    input_file="recording.mdf",
    output_format="HDF5"  # Convert MDF to HDF5
)
```

**Features:**
- **Native MDF reading**: Reads .mdf files through the `MCSX.Data` COM interface
- **Metadata access**: `MDFFileReader.get_metadata()` returns acquisition parameters (pixel size, magnification, frame timing, channel names)
- **Channel selection**: The `channel_idx` option selects which channels to read (1-based)
- **Windows only**: Requires the `pywin32` package and the `MCSX.Data` COM server (installed with the MDF acquisition software); `NotImplementedError` is raised otherwise
- **Read-only**: No MDF writer (convert to HDF5 or TIFF for output)

## Output Format Reference

`output_format` values accepted by `get_video_file_writer` and `OFOptions`:

| Format string | Output |
|---------------|--------|
| `TIFF` | Single multi-page TIFF file |
| `HDF5` | Single HDF5 file, one 3D dataset per channel |
| `MAT` | MATLAB MAT-file, one 3D array per channel (`OFOptions` default) |
| `SUITE2P_TIFF` | TIFF with channels written as interleaved single-channel pages |
| `MULTIFILE_TIFF` | One TIFF file per channel |
| `MULTIFILE_MAT` | One MAT-file per channel |
| `MULTIFILE_HDF5` | One HDF5 file per channel |
| `CAIMAN_HDF5` | One HDF5 file per channel with dataset name `/mov` (used by CaImAn) |
| `ARRAY` | In-memory accumulation (`ArrayWriter`), retrieved with `get_array()` |
| `NULL` | Frames are discarded (callback-only processing) |
| `BEGONIA` | Not yet implemented (raises `NotImplementedError`) |

The memory formats `ARRAY` and `NULL` ignore the file path and `output_path`. Note that `compensate_arr` always returns arrays regardless of `options.output_format`; see the [motion correction API reference](../api/motion_correction.md).

### In-Memory Output

The `ARRAY` and `NULL` writers can be constructed directly through the writer factory; the file path argument is ignored:

```{literalinclude} ../snippets/user_guide/file_formats/memory_writers.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Multi-File Formats

Split outputs across one file per channel:

```python
options = OFOptions(
    input_file="large_video.h5",
    output_format="MULTIFILE_HDF5"  # or MULTIFILE_TIFF, MULTIFILE_MAT
)
```

Files are named `<basename>_ch<N>.<FORMAT>`, e.g. `compensated_ch1.HDF5`, `compensated_ch2.HDF5`. If the output filename has no extension, it is treated as a directory and the basename `compensated` is used.

The multi-file writers can also be used directly through the writer factory; the per-channel files are created on the first write:

```{literalinclude} ../snippets/user_guide/file_formats/multifile_writer.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### CaImAn Output

```python
options = OFOptions(
    output_format="CAIMAN_HDF5"  # Per-channel HDF5 files with /mov dataset
)
```

Writes per-channel HDF5 files using the dataset name `/mov` expected by CaImAn.

### Suite2p-Style TIFF

```python
options = OFOptions(
    output_format="SUITE2P_TIFF"
)
```

Writes a TIFF where channels are stored as interleaved single-channel pages.

## Factory Functions

### Creating Readers

```{literalinclude} ../snippets/user_guide/file_formats/factory_readers.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Creating Writers

```{literalinclude} ../snippets/user_guide/file_formats/factory_writer.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## Configuration Options

### Buffer Size and Binning

```python
options = OFOptions(
    buffer_size=400,  # Number of frames per batch (default: 400)
    bin_size=1  # Temporal binning factor (default: 1)
)
```

- `buffer_size`: Number of frames read per batch; larger buffers use more memory per batch
- `bin_size`: Temporal binning applied during reading (bin_size=2 averages every 2 consecutive frames)

### Output Data Type

```python
options = OFOptions(
    output_typename="double"  # Default
)
```

Recognized values: `"double"` (float64), `"single"` (float32), `"uint8"`, `"uint16"`, `"int16"`, `"int32"`. The cast is currently applied only by `compensate_arr` when returning arrays; file-based writers infer the dtype from the frames they receive.

### Saving Displacement Fields

```python
options = OFOptions(
    save_w=True,  # Save displacement fields (default: False)
    output_format="HDF5"
)
```

With file-based output, displacement fields are written to `w.h5` in `output_path` as an HDF5 file with two 3D datasets: `u` (horizontal, x) and `v` (vertical, y).

### File Naming

```python
options = OFOptions(
    naming_convention="default",  # or "batch"
    output_file_name="my_output.h5"  # Optional custom name
)
```

- `"default"`: `compensated.<FORMAT>` (e.g. `compensated.HDF5`)
- `"batch"`: `<input_basename>_compensated.<FORMAT>`
- Custom: Override with `output_file_name`

## Reading Data

### Array-Like Indexing

All readers support numpy-style indexing:

```{literalinclude} ../snippets/user_guide/file_formats/array_indexing.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

With `bin_size > 1`, indices refer to binned frames: `reader[0]` returns the average of the first `bin_size` raw frames.

### Batch Iteration

```{literalinclude} ../snippets/user_guide/file_formats/batch_iteration.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Multi-Channel from Separate Files

```{literalinclude} ../snippets/user_guide/file_formats/multichannel_reader.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

All files must have matching resolution and frame count; their channels are concatenated into a single `(T, H, W, C_total)` stream. Passing a list of paths to `get_video_file_reader()` creates the same reader.

## Writing Data

### Batch Writing

```{literalinclude} ../snippets/user_guide/file_formats/batch_writing.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Context Manager

```{literalinclude} ../snippets/user_guide/file_formats/writer_context_manager.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## Format Conversion

### Simple Conversion

```{literalinclude} ../snippets/user_guide/file_formats/tiff_to_hdf5.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Batch Conversion

```{literalinclude} ../snippets/user_guide/file_formats/batch_conversion.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## Performance Tips

### HDF5 Optimization

```{literalinclude} ../snippets/user_guide/file_formats/hdf5_writer_options.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Memory-Mapped TIFF

```{literalinclude} ../snippets/user_guide/file_formats/tiff_memmap.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

### Buffer Size Tuning

```python
# Larger buffers read more frames per batch and use more memory;
# smaller buffers reduce memory usage at the cost of more read operations.

# For large RAM:
options = OFOptions(buffer_size=1000)

# For limited RAM:
options = OFOptions(buffer_size=100)
```
