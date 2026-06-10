# Parameter Guide

This guide explains the key parameters in PyFlowReg's variational optical flow implementation and provides guidance on their recommended values based on the characteristics of 2-photon microscopy data.

## Diffusion Parameters

The optical flow solver uses non-linear diffusion regularization with two key parameters that control the smoothness of the computed displacement fields.

### `a_smooth`: Smoothness Diffusivity

**Default value**: `1.0`

The `a_smooth` parameter controls the diffusivity for the smoothness term in the variational optical flow model. It determines how strongly neighboring displacement values influence each other during the optimization process.

**Recommended value for 2-photon microscopy**: `a_smooth = 1.0` (linear diffusion)

**Justification**:
In 2-photon microscopy of neural tissue, we typically observe continuous, smooth motion patterns without sharp discontinuities. Biological samples undergo smooth deformations (breathing artifacts, tissue drift, thermal expansion) rather than abrupt jumps. Using linear diffusion (`a_smooth = 1.0`) is appropriate because:

- Neural tissue moves coherently - neighboring pixels experience similar displacements
- No sharp motion boundaries are expected within the field of view
- Linear diffusion provides adequate smoothing while preserving fine motion details
- Computational efficiency is maintained with the linear formulation

### `a_data`: Data Term Diffusivity

**Default value**: `0.45`

The `a_data` parameter controls the diffusivity for the data term, which measures how well the optical flow model fits the observed brightness changes in the video.

**Recommended value**: `a_data = 0.45` (sublinear diffusion)

**Justification**:
The sublinear value follows best practices from {cite}`sun2010secrets` for handling outliers and noise in optical flow estimation:

- **Robustness to noise**: 2-photon videos contain significant photon shot noise, especially in dim regions. Sublinear diffusion (`a_data < 1`) reduces the influence of large residuals caused by noise
- **Outlier handling**: Occasional bright spots from autofluorescence or imaging artifacts are downweighted, preventing them from distorting the flow field
- **Edge preservation**: Sublinear diffusion allows the model to handle brightness discontinuities at cell boundaries more gracefully
- **Empirical validation**: This value has been validated across diverse 2-photon imaging datasets {cite}`flotho2022flow`

### Graduated Non-Convexity: `gnc_schedule`

**Default value**: `None` (disabled)

Optional graduated non-convexity (GNC) staging can be enabled with
`gnc_schedule`, for example `(0.0, 0.5, 1.0)`. The schedule is a sequence of
stage weights that must be nondecreasing, start at `0.0`, and end at `1.0`.
Each stage reruns the full pyramid, initialized with the flow from the
previous stage, and blends the quadratic and robust penalty weights with the
fixed stage weight `beta`: the data-term non-linearity becomes
`(1 - beta) + beta * robust` whenever `a_data < 1` (and analogously for the
smoothness term when `0 < a_smooth < 1`). The first stage (`beta = 0.0`) thus
solves a quadratic problem, the final stage (`beta = 1.0`) the fully robust
one; the configured `a_data` and `a_smooth` values themselves are unchanged.
In GNC mode each pyramid level additionally performs repeated
warp/relinearize steps, controlled by `warping_steps` (10 per level if not
set). Leaving `gnc_schedule=None` preserves the single-pass solver path.
See [Data Terms](data_terms.md) for the full treatment of the penalty
functions.

## Spatial-Temporal Filtering: `sigma`

The `sigma` parameter controls Gaussian filtering applied to the video data before optical flow computation. It is specified as `[σx, σy, σt]` for each channel, where:

- `σx`, `σy`: Spatial filtering in x and y dimensions (pixels)
- `σt`: Temporal filtering across frames

**Default**: `[[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]` for 2-channel data

### Advantages of 3D Filtering

**Noise reduction**:
- Gaussian filtering in x, y, and t reduces photon shot noise while preserving motion information
- Temporal filtering (`σt > 0`) is particularly effective for smoothing frame-to-frame intensity fluctuations

**Improved gradient estimation**:
- Spatial derivatives needed for optical flow are more stable when computed from filtered data
- Temporal derivatives benefit from smoothing across adjacent frames

**Motion coherence**:
- Small `σt` values (e.g., 0.1-0.5 frames) help enforce temporal consistency without excessive motion blur

### Disadvantages of 3D Filtering

**Temporal blur**:
- Excessive temporal filtering (`σt` too large) blurs rapid motion events
- Fast transient signals (e.g., calcium spikes) may be attenuated
- Trade-off between noise reduction and temporal resolution

**Increased computation**:
- 3D Gaussian filtering is more computationally expensive than 2D
- Requires buffering multiple frames in memory

**Parameter tuning**:
- Optimal `σt` depends on frame rate and motion speed
- Different channels may require different filtering strengths
- May need adjustment for datasets with varying noise characteristics

### Recommended Settings

For typical 2-photon calcium imaging:
```{literalinclude} ../snippets/theory/parameters/sigma_calcium.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

For high frame rate imaging with low SNR:
```{literalinclude} ../snippets/theory/parameters/sigma_low_snr.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

For volumetric imaging (z-stacks):
```{literalinclude} ../snippets/theory/parameters/sigma_volumetric.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

## References

```{bibliography}
:filter: docname in docnames
```
