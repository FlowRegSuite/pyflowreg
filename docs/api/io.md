# I/O System

Modular I/O system supporting multiple file formats through a common VideoReader/VideoWriter interface.

## Factory Functions

`get_video_file_reader()` and `get_video_file_writer()` select the appropriate reader or writer for a given file path or array and are the recommended entry points for video I/O.

```{eval-rst}
.. automodule:: pyflowreg.util.io.factory
   :members:
```

## Base Classes

`VideoReader` returns data in `(T, H, W, C)` format and supports both array-like indexing (`reader[10:20]`, with automatic temporal binning) and sequential batch reading via `read_batch()` / `has_batch()`, which is how `compensate_recording` consumes input files.

```{eval-rst}
.. automodule:: pyflowreg.util.io._base
   :members:
   :show-inheritance:
```

## Dataset Discovery

```{eval-rst}
.. automodule:: pyflowreg.util.io._ds_io
   :members:
   :show-inheritance:
```

## File Format Support

### HDF5

```{eval-rst}
.. automodule:: pyflowreg.util.io.hdf5
   :members:
   :show-inheritance:
```

### TIFF

```{eval-rst}
.. automodule:: pyflowreg.util.io.tiff
   :members:
   :show-inheritance:
```

### MATLAB MAT

```{eval-rst}
.. automodule:: pyflowreg.util.io.mat
   :members:
   :show-inheritance:
```

### MDF (Sutter)

Reading MDF files requires Windows, the `pywin32` package, and the `MCSX.Data` COM server. See [Supported Formats](../user_guide/file_formats.md) for details.

```{eval-rst}
.. automodule:: pyflowreg.util.io.mdf
   :members:
   :show-inheritance:
```

## Multi-File Handling

```{eval-rst}
.. automodule:: pyflowreg.util.io.multifile_wrappers
   :members:
   :show-inheritance:
```

## Array I/O

```{eval-rst}
.. automodule:: pyflowreg.util.io._arr
   :members:
   :show-inheritance:
```

## Null Writer (Callback-Only Processing)

```{eval-rst}
.. automodule:: pyflowreg.util.io._null
   :members:
   :show-inheritance:
```

## ScanImage Support

```{eval-rst}
.. automodule:: pyflowreg.util.io._scanimage
   :members:
```
