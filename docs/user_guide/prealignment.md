# Cross-Correlation Pre-Alignment

Pre-alignment is an optional per-frame step that estimates a single global
translation between each frame and the reference using phase cross-correlation
before the variational solver runs. The estimated shift is added to the flow
initialization as a constant displacement, so the variational solver only needs
to recover the remaining (non-rigid) motion. This is useful when recordings
contain translations that are large relative to what the flow initialization
and pyramid already absorb, for example slow stage or sample drift.

## How It Works

For each frame, with the current flow initialization `w_init`:

1. The preprocessed frame is backward-warped by `w_init`.
2. `estimate_rigid_xcorr_2d` (in `pyflowreg.util.xcorr_prealignment`) estimates
   the residual global translation between the preprocessed reference and the
   partially aligned frame.
3. The translation is added to `w_init` as a constant `(u, v)` shift.
4. The preprocessed frame is backward-warped by the combined field, and the
   variational solver is run from a zero initialization to recover the
   non-rigid residual.
5. The final displacement field is the sum of initialization, rigid shift, and
   non-rigid residual; the raw frame is warped once with this total field.

Inside `estimate_rigid_xcorr_2d`:

- Multi-channel images are reduced to a single channel using the normalized
  channel weights, or the plain channel mean if no weights are given.
- Both images are downsampled to at most `cc_hw` pixels per dimension (the
  target size is capped at the image size, so images are never upsampled),
  mean-subtracted, and multiplied with a Hann window to suppress periodic
  boundary artifacts.
- The shift is computed with scikit-image's `phase_cross_correlation` using
  phase normalization and sign disambiguation, with `cc_up` as the subpixel
  upsampling factor.
- The result is rescaled to the full-resolution grid and negated to match the
  backward-warp convention, yielding a constant displacement `(u, v)` where
  `u` is the horizontal (x) and `v` the vertical (y) component.

## Enabling Pre-Alignment

```{literalinclude} ../snippets/user_guide/prealignment/enable_prealignment.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

The same options work with `compensate_arr` for in-memory arrays.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cc_initialization` | `False` | Enable cross-correlation pre-alignment. |
| `cc_hw` | `256` | Target size for the correlation, as a single int or an `(height, width)` tuple. Images are downsampled to this size before correlation; values larger than the image are capped at the image size. |
| `cc_up` | `1` | Upsampling factor passed to `phase_cross_correlation` for subpixel accuracy. `1` resolves shifts to whole pixels on the downsampled grid. |

Because the correlation runs on the downsampled images, the estimated shift at
`cc_up=1` is quantized to one pixel of the reduced grid, which corresponds to
`H / cc_hw` pixels at full resolution (e.g. 2 pixels for a 512-pixel image
correlated at 256). Increase `cc_up` for finer estimates, or `cc_hw` to reduce
the downsampling. The non-rigid solver runs afterwards in any case and refines
the result.

## Scope and Limitations

- The estimate is a single global translation per frame; rotation, scaling,
  and local deformation are left to the variational solver.
- Pre-alignment runs on the batch-processing executor path used by
  `compensate_recording` and `compensate_arr`. All three executors
  (sequential, threading, multiprocessing) implement it; see
  [Parallelization](parallelization.md) for executor selection.
- Reference preregistration in `OFOptions.get_reference_frame` forwards the
  `cc_*` settings, so an enabled pre-alignment also applies while building
  the reference. (The MATLAB reference does not pre-align its
  preregistration; this is a deliberate improvement.)
- It is **not** applied by:
  - Direct `get_displacement` calls — the flow backends do not accept the
    `cc_*` parameters; the executors consume them before invoking the solver.
  - `FlowRegLive` online processing, which has no pre-alignment handling.
- Each pre-aligned frame requires two additional warps and one FFT-based
  correlation, so leave it disabled when frame-to-frame motion is small.

The multi-session pipeline uses the same `estimate_rigid_xcorr_2d` function
separately to rigidly align session averages; see
[Multi-Session Processing](multi_session.md).

## See Also

- [Configuration](configuration.md) - Complete `OFOptions` parameter reference
- [Parallelization](parallelization.md) - Executor selection and constraints
- [Utilities API](../api/utilities.md) - `estimate_rigid_xcorr_2d` reference
