# Troubleshooting

This page collects errors and warnings that PyFlowReg raises in practice,
with the cause and the fix for each. Every message below is quoted from the
source as currently implemented. If a symptom you are seeing is not listed
here, check the relevant page in the user guide or open an issue.

## Installation and import issues

### `tifffile not installed. TIFF support unavailable.`

This warning is emitted at import time when the `tifffile` package is not
installed. TIFF reading and writing are unavailable until it is installed.
Attempting to construct a TIFF reader or writer afterwards raises:

```
ImportError: tifffile library required for TIFF support
```

The same `ImportError` is raised by the ScanImage reader path. Install
`tifffile` to restore TIFF support.

### `Multispectral mapping with >3 channels requires 'scikit-learn' library`

Raised by the visualization helpers in `pyflowreg.util.visualization` when an
image with more than three channels is passed to the color mapping routine and
scikit-learn (used for PCA reduction to three components) is not installed.
Install scikit-learn, or reduce the data to at most three channels before
visualization.

### `Matplotlib backend requires 'matplotlib' library`

Raised by `quiver_visualization(...)` when called with `backend="matplotlib"`
and matplotlib is not installed. The `backend="opencv"` path does not require
matplotlib, so either install matplotlib or switch to the OpenCV backend.

### MDF (Sutter) files on non-Windows systems

Constructing an MDF reader without pywin32 raises:

```
NotImplementedError: MDF file reading requires Windows and 'pywin32' library
```

MDF reading uses the `MCSX.Data` COM server and is Windows-only. There is no
cross-platform fallback. On Windows, install pywin32 and the MDF software that
provides the COM server. On other platforms, convert the data to a supported
format (TIFF or HDF5) first.

### A flow backend is missing from `list_backends()`

The optional flow backends register themselves only when their dependency is
importable:

- `diso` requires OpenCV (`cv2`).
- `flowreg_torch` requires PyTorch (`torch`).
- `flowreg_cuda` requires CuPy (`cupy`).

When the dependency is absent the backend is simply not registered: it does not
appear in `list_backends()` and selecting it raises the unknown-backend
`ValueError` described under [Backend and executor problems](#backend-and-executor-problems).
Install the corresponding dependency to make the backend available. See
[user_guide/backends.md](user_guide/backends.md) for the backend list and
[user_guide/parallelization.md](user_guide/parallelization.md) for which
executors each backend supports.

## Configuration errors

`OFOptions` and the z-alignment config (`ZAlignConfig`) are Pydantic v2 models
configured with `extra="forbid"`, so passing a field name they do not define
raises a Pydantic validation error reporting that extra fields are not
permitted. Check the field name against
[user_guide/configuration.md](user_guide/configuration.md). (The session
`SessionConfig` does not forbid extra fields.)

### `Alpha must be positive` / `All alpha values must be positive` / `Alpha must be scalar or 2-element tuple`

Raised by the `alpha` validator in `OFOptions`. `alpha` must be a positive
scalar (expanded internally to `(alpha, alpha)`) or a 2-element tuple of
positive values. Negative, zero, or wrongly shaped values trigger these
errors.

### `1D sigma must be [sx, sy, st]` / `2D sigma must be (n_channels, 3)` / `Sigma must be [sx,sy,st] or (n_channels, 3)`

Raised by the `sigma` validator in `OFOptions`. The Gaussian smoothing
parameter must be either a length-3 sequence `[sx, sy, st]` (applied to all
channels) or a 2D array shaped `(n_channels, 3)`. Any other shape raises one of
these errors.

### GNC schedule validation

The `gnc_schedule` validator in `OFOptions` enforces several conditions, each
with its own message:

- `gnc_schedule must be a 1D sequence`
- `gnc_schedule must contain at least two stages`
- `gnc_schedule entries must lie in [0, 1]`
- `gnc_schedule must be monotone nondecreasing`
- `gnc_schedule must start at 0.0`
- `gnc_schedule must end at 1.0`

A valid schedule is a 1D sequence of at least two values in `[0, 1]` that is
non-decreasing, starts at `0.0`, and ends at `1.0`, for example
`(0.0, 0.5, 1.0)`. See [theory/parameters.md](theory/parameters.md) for what
GNC staging does.

### Invalid constancy assumption

`constancy_assumption` accepts the serialized values `'gc'` (gradient
constancy, the default), `'gray'` (brightness constancy), and `'cs'` (census
constancy); the aliases `'gradient'`, `'brightness'`, and `'census'` are
normalized to these. An unrecognized value fails Pydantic enum validation when
the option is constructed. If an unrecognized value reaches the solver it
raises:

```
ValueError: Unknown constancy assumption: '<value>'. Supported values are: '...'.
```

See [theory/data_terms.md](theory/data_terms.md) for the data terms.

### z-alignment config validation

`ZAlignConfig` validators enforce, among others:

- `Value must be >= 1` for the positive-integer fields.
- `Value must be > 0` for `stage1_alpha`, the sigma fields, and
  `parabolic_tau_scale`.
- `overlap must satisfy 0 <= overlap < 1`.
- `z_shift_file must have an HDF5 extension (.h5/.hdf5/.hdf), got: <name>` —
  the z-shift file is written and re-read with the HDF5 reader, so a non-HDF5
  extension is rejected.
- `Invalid output_dtype: <value>` when `output_dtype` is not a valid NumPy
  dtype name.
- `n_jobs must be -1 or >= 1`.
- `parallelization must be one of ['sequential', 'threading']` — z-alignment
  supports only these two executors.

See [user_guide/z_align.md](user_guide/z_align.md).

## Backend and executor problems

### `Unknown flow backend: '<name>'. Available backends: [...]`

Raised by `get_backend()` (and during `OFOptions.resolve_get_displacement()`)
when `flow_backend` names a backend that is not registered. The message lists
the backends that are registered in the current environment. If you expected a
backend such as `diso`, `flowreg_torch`, or `flowreg_cuda` to be present,
install its dependency (see
[Installation and import issues](#installation-and-import-issues)).

### `diso` backend combined with census/gray constancy or GNC

`OFOptions.resolve_get_displacement()` guards the `diso` backend, which only
supports the gradient-constancy data term and does not support graduated
non-convexity:

```
ValueError: The 'diso' backend does not support variational constancy
assumption '<value>'. Use flow_backend='flowreg' for 'gray' or 'cs'.
```

```
ValueError: The 'diso' backend does not support graduated non-convexity.
Use flow_backend='flowreg' for 'gnc_schedule' or 'warping_steps'.
```

Use the default `flow_backend='flowreg'` if you need a non-`'gc'` constancy
assumption, a `gnc_schedule`, or `warping_steps`.

### Executor fallback warning

When the requested executor is not supported by the selected backend or is not
available on the system, `compensate_recording` selects an alternative and
warns, for example:

```
Backend '<backend>' does not support '<executor>' executor. Supported
executors: [...]. Falling back to '<fallback>'.
```

or, when the executor is unavailable on the host:

```
Executor '<executor>' is not available on this system. Supported executors:
[...]. Falling back to '<fallback>'.
```

This is informational; processing continues with the fallback executor. The
canonical description of which backend supports which executor (including the
GPU-backend constraints) lives in
[user_guide/parallelization.md](user_guide/parallelization.md).

## I/O problems

These messages come from the reader/writer factories in
`pyflowreg.util.io.factory`.

### `File not found: <path>`

`FileNotFoundError` raised by `get_video_file_reader()` when the given path
does not exist. Check the path; remember that the repository root folder name
may contain a space, so quote paths in shell commands.

### `Unsupported file format: <ext>`

Raised by `get_video_file_reader()` when the file extension is not one of the
recognized formats (`.tif`/`.tiff`, `.h5`/`.hdf5`/`.hdf`, `.mat`, `.mdf`) and
the file could not be opened as HDF5. When the HDF5 probe fails, a warning is
emitted first:

```
File format detection failed: could not open as HDF5: <error>
```

Convert the input to a supported format, or pass a list of per-channel files
for multichannel input. See [user_guide/file_formats.md](user_guide/file_formats.md).

### `Folder <path> does not contain images` / image-folder reading

Passing a directory to `get_video_file_reader()` is only partially supported.
If the folder contains image files, you get:

```
NotImplementedError: Image folder reading not yet implemented. Use TIFF
stacks instead.
```

If the folder contains no images, you get:

```
ValueError: Folder <path> does not contain images
```

Combine the images into a TIFF stack (or another supported format) and read
that instead.

### `BEGONIA format not yet implemented`

`NotImplementedError` raised by `get_video_file_writer()` when
`output_format='BEGONIA'`. This output format is not implemented; choose one of
the supported output formats listed in
[user_guide/file_formats.md](user_guide/file_formats.md).

### `Unsupported output format: <value>`

Raised by `get_video_file_writer()` when `output_format` is not one of the
recognized strings. Use a supported value (for example `'TIFF'`, `'HDF5'`,
`'MAT'`, `'ARRAY'`, or `'NULL'`).

## Quality problems (not errors)

These are not error conditions, but symptoms that point to parameter or
reference choices.

- **Residual motion / under-registered output.** Review the regularization and
  iteration settings. The recommended values and the trade-offs around
  `a_data`, `a_smooth`, and `sigma` are documented in
  [theory/parameters.md](theory/parameters.md); do not change them blindly.
- **Poor results traced to the reference frame.** A noisy or mis-chosen
  reference degrades the whole sequence. See
  [user_guide/prealignment.md](user_guide/prealignment.md) for building a
  robust, prealigned reference.
