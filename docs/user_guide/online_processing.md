# Online Processing

`FlowRegLive` provides frame-by-frame motion correction for streaming scenarios where frames arrive one at a time, such as live previews during acquisition or closed-loop experiments. Unlike the batch APIs (`compensate_arr`, `compensate_recording`), which process a whole recording against a fixed reference, `FlowRegLive` keeps internal state between calls: a preregistered reference that is periodically updated, a temporal filter buffer, and the previous displacement field used to initialize the next flow computation.

The full class API is documented in the [motion correction API reference](../api/motion_correction.md).

## Creating a Corrector

```{literalinclude} ../snippets/user_guide/online_processing/create_corrector.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `options` | `None` | `OFOptions` instance. If `None`, options are created with fast defaults. |
| `reference_buffer_size` | `50` | Capacity of the circular buffer that collects incoming frames before a reference is set. |
| `reference_update_interval` | `20` | Every N-th processed frame is blended into the reference. |
| `reference_update_weight` | `0.2` | Blend weight: `ref = (1 - w) * ref + w * warped_frame`. |
| `truncate` | `4.0` | Gaussian filters are truncated at this many standard deviations. |

Additional keyword arguments are forwarded as `OFOptions` overrides.

Note that `FlowRegLive` always overrides `quality_setting` to `"fast"`, even when you pass your own `OFOptions`. With `min_level` left at its default of `-1`, the fast preset stops pyramid refinement at level 6 instead of descending to the full-resolution level 0, trading accuracy for per-frame latency. Other options (`alpha`, `sigma`, `levels`, `iterations`, `eta`, `channel_normalization`, `interpolation_method`, flow backend selection) are taken from the provided options.

## Setting the Reference

A reference must be set before correction starts. `set_reference()` accepts an optional array:

```python
# From an explicit frame stack (T, H, W, C): frames are preregistered first
flow_reg.set_reference(video[100:201])

# From a single frame (H, W) or (H, W, C): used directly
flow_reg.set_reference(video[0])

# From the internal buffer (frames passed to __call__ before a reference existed)
flow_reg.set_reference()
```

Behavior by input shape:

- **4D stack `(T, H, W, C)`** -- the frames are preregistered with `compensate_arr` against their temporal mean, using a copy of the options with `quality_setting="balanced"`. The mean of the registered frames becomes the reference.
- **3D `(H, W, C)` or 2D `(H, W)`** -- used directly as a single-frame reference, without preregistration. A 3D input is always interpreted as one multi-channel frame, so a grayscale stack `(T, H, W)` must be given an explicit channel axis first (e.g. `frames[..., None]`).
- **No argument** -- the frames accumulated in the internal reference buffer are stacked and processed as above. Raises `ValueError` if the buffer is empty.

After the reference is established, it is normalized and spatially Gaussian-filtered; the min/max of this filtered reference is stored and used to normalize every incoming frame (per channel when `channel_normalization="separate"`, jointly otherwise). Setting a reference also resets the stored flow initialization and clears the temporal filter buffer.

`reset_reference(new_reference)` is an alias that calls `set_reference()` with the given array.

## Processing Frames

`FlowRegLive` is callable. Each call takes one frame and returns the corrected frame and its displacement field:

```python
registered, flow = flow_reg(frame)
```

- `frame`: input frame, `(H, W, C)` or `(H, W)`.
- `registered`: corrected frame, `(H, W, C)`.
- `flow`: displacement field, `(H, W, 2)`, where `flow[..., 0]` is `u` (horizontal, x) and `flow[..., 1]` is `v` (vertical, y).

If no reference has been set yet, the call instead appends the frame to the reference buffer and returns the input unchanged together with an all-zero flow field. This allows streaming frames from the start and calling `set_reference()` once enough frames have been collected.

Once a reference exists, each call:

1. Normalizes the frame using the stored reference min/max values.
2. Applies the 2D spatial Gaussian filter and pushes the result into the temporal buffer.
3. Applies the causal temporal half-kernel filter (see below).
4. Computes the displacement field against the filtered reference, initialized with the previous frame's flow.
5. Warps the original (unfiltered) input frame with the resulting field, using `options.interpolation_method` (default cubic).
6. Every `reference_update_interval`-th frame, blends the warped frame into the reference with weight `reference_update_weight` and refreshes the normalization min/max from the updated reference.

The `normalize` keyword of `__call__` is currently unused; normalization is always applied.

Convenience methods:

- `register_frames(frames)` -- runs a `(T, H, W, C)` stack through `__call__` sequentially and returns `(registered_frames, flow_fields)` with shapes `(T, H, W, C)` and `(T, H, W, 2)`.
- `get_current_flow()` -- returns a copy of the last displacement field, or `None`.
- `set_flow_init(w_init)` -- overrides the flow initialization used for the next frame.

## Spatial and Temporal Filtering

Batch processing filters each batch with a 3D Gaussian over `(y, x, t)`. In a streaming setting future frames are not available, so `FlowRegLive` splits the filter:

1. Each frame is filtered in 2D with the spatial components of `options.sigma` (`pyflowreg.util.image_processing.apply_gaussian_filter`).
2. The 2D-filtered frames are kept in a circular buffer of size `max(1, int(truncate * sigma_t + 0.5) + 1)`, where `sigma_t` is the temporal sigma.
3. The frame used for flow estimation is a weighted average of the current and past buffered frames with normalized half-Gaussian weights (`pyflowreg.util.image_processing.gaussian_filter_1d_half_kernel`). Only the current and past frames contribute, so the filter is causal and adds no latency, but it is not identical to the symmetric temporal filtering used in batch mode.

With the `OFOptions` default `sigma` of `[[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]`, the temporal sigma of 0.1 yields a buffer of size 1 and no effective temporal filtering; increase the third sigma component to enable it. When per-channel sigmas are configured, `FlowRegLive` currently uses the first channel's spatial sigmas for all channels and the maximum temporal sigma across channels.

## Complete Example

The following streams a short recording (`recording.tif`) through the corrector frame by frame; `examples/jupiter_demo_live.py` runs the same loop on the Jupiter demo recording and additionally shows an interactive side-by-side display:

```{literalinclude} ../snippets/user_guide/online_processing/streaming_loop.py
:language: python
:start-after: "[docs:start]"
:end-before: "[docs:end]"
```

In a real acquisition loop, replace the `for` loop body with frames delivered by your acquisition system; `FlowRegLive` only requires one `(H, W, C)` or `(H, W)` array per call.
