# 3D Z-Stack Alignment

This guide covers stack alignment for 3D volumetric data acquired through z-stack scanning in 2-photon microscopy.

PyFlowReg performs **2D frame-by-frame motion correction**. It does not perform true 3D volumetric registration. For z-stack alignment, PyFlowReg uses an adaptive reference approach where the reference frame is updated slice-by-slice as you move through the stack.

## Z-Stack Acquisition Strategy

When acquiring z-stacks for motion correction, use this acquisition pattern:

1. **Multiple frames per z-slice**: Acquire multiple repetitions (e.g., 5-10 frames) at each z-position
2. **Sequential scanning**: Move through z-positions sequentially, acquiring all repetitions before moving to next z-plane
3. **Save as single file**: Store the complete acquisition as a single multi-frame file

Example acquisition pattern for 3 z-slices with 9 frames each:
```
Frame 1-9:   Z-position 1 (9 repetitions)
Frame 10-18: Z-position 2 (9 repetitions)
Frame 19-27: Z-position 3 (9 repetitions)
```

## Alignment Workflow

The alignment uses adaptive reference updating to handle the changing image content across z-depths:

### Python Example

```python
from pyflowreg.motion_correction import compensate_recording, OFOptions
from pyflowreg.util.io.factory import get_video_file_reader
import numpy as np

# Step 1: Register with adaptive reference
frames_per_slice = 9  # Number of repetitions per z-position

options = OFOptions(
    input_file="stack.tif",
    output_path="aligned_sequence/",
    output_format="HDF5",
    quality_setting="fast",
    alpha=25.0,
    buffer_size=frames_per_slice,  # Process one complete z-slice at a time
    update_reference=True,  # Update reference after each slice
    reference_frames=list(range(frames_per_slice)),  # Use first slice as initial reference
    bin_size=1,
)

compensate_recording(options)

# Step 2: Read back and bin the registered frames
reader = get_video_file_reader(
    "aligned_sequence/compensated.HDF5",
    buffer_size=100,
    bin_size=frames_per_slice  # Bin the repetitions for each slice
)

# Read all binned slices
volume = []
while reader.has_batch():
    batch = reader.read_batch()
    volume.append(batch)

volume = np.concatenate(volume, axis=0)  # (Z, H, W, C)

# Save final aligned and binned volume
np.save("aligned_volume.npy", volume)
```

## How It Works

### Adaptive Reference Strategy

1. **Initial reference**: The first z-slice (frames 1-9) is used as the initial reference
2. **First slice registration**: Frames 1-9 are registered to their average
3. **Reference update**: After processing the first slice, the reference is updated to the registered average of frames 1-9
4. **Next slice**: Frames 10-18 are registered to the updated reference (which is similar to them since it's the previous z-plane)
5. **Repeat**: This continues through all z-slices

This approach works because:
- Adjacent z-slices have similar image content
- Each new slice is registered to the previous slice, not to a distant reference
- Slowly appearing structures (deeper tissue features) don't cause registration failures
- Motion correction adapts to the changing anatomy as you move through depth

### Key Parameters

**buffer_size**: Must equal frames_per_slice
- Controls how many frames are processed before updating the reference
- Ensures reference updates happen between z-slices, not within a slice

**update_reference**: Must be True
- Enables the adaptive reference strategy
- Without this, all frames would be registered to the first slice only

**reference_frames**: First slice indices (e.g., [0, 1, 2, ..., 8] for 9 frames)
- Defines which frames form the initial reference
- Should span the first complete z-slice

**alpha**: Typically 20-50 for z-stacks
- Lower values allow more flexible registration between different z-depths
- Higher values enforce smoother, more constrained motion

## Post-Processing: Frame Binning

After registration, bin (average) the repeated frames for each z-slice:

```python
from pyflowreg.util.io.factory import get_video_file_reader

# The reader can perform binning automatically
reader = get_video_file_reader(
    "aligned_sequence/compensated.HDF5",
    buffer_size=100,
    bin_size=9  # Bin every 9 frames
)

# This returns already-binned data where each "frame" is the average of 9 registered frames
binned_volume = []
while reader.has_batch():
    batch = reader.read_batch()
    binned_volume.append(batch)

binned_volume = np.concatenate(binned_volume, axis=0)
```

Alternatively, manual binning:

```python
import numpy as np

# Load registered frames
registered = np.load("registered_frames.npy")  # (T, H, W, C) where T = Z * frames_per_slice

frames_per_slice = 9
n_slices = registered.shape[0] // frames_per_slice

# Reshape and average
volume = registered.reshape(n_slices, frames_per_slice, H, W, C)
volume = np.mean(volume, axis=1)  # (Z, H, W, C)
```

## Complete Pipeline Example

```python
import numpy as np
from pathlib import Path
from pyflowreg.motion_correction import compensate_recording, OFOptions
from pyflowreg.util.io.factory import get_video_file_reader

def align_zstack(
    input_file,
    output_dir,
    frames_per_slice=9,
    alpha=25.0,
    quality="fast"
):
    """
    Complete z-stack alignment pipeline.

    Parameters
    ----------
    input_file : str
        Path to multi-frame z-stack (e.g., TIFF with repeated frames per slice)
    output_dir : str
        Output directory for results
    frames_per_slice : int
        Number of frame repetitions acquired at each z-position
    alpha : float
        Regularization strength (20-50 recommended for z-stacks)
    quality : str
        Quality setting: "fast", "balanced", or "quality"

    Returns
    -------
    volume : np.ndarray
        Aligned and binned z-stack (Z, H, W, C)
    """

    # Step 1: Configure alignment
    options = OFOptions(
        input_file=input_file,
        output_path=output_dir,
        output_format="HDF5",
        quality_setting=quality,
        alpha=alpha,
        buffer_size=frames_per_slice,
        update_reference=True,
        reference_frames=list(range(frames_per_slice)),
        save_w=True,  # Save displacement fields for QC
        save_meta_info=True,
    )

    # Step 2: Run registration
    print(f"Registering z-stack with {frames_per_slice} frames per slice...")
    compensate_recording(options)

    # Step 3: Load and bin registered frames
    print("Binning registered frames...")
    output_file = Path(output_dir) / "compensated.HDF5"

    reader = get_video_file_reader(
        str(output_file),
        buffer_size=100,
        bin_size=frames_per_slice
    )

    volume = []
    while reader.has_batch():
        batch = reader.read_batch()
        volume.append(batch)

    volume = np.concatenate(volume, axis=0)

    print(f"Final volume shape: {volume.shape}")

    # Step 4: Save final volume
    output_volume = Path(output_dir) / "aligned_volume.npy"
    np.save(output_volume, volume)
    print(f"Saved to {output_volume}")

    return volume

# Usage
volume = align_zstack(
    input_file="my_stack.tif",
    output_dir="results/",
    frames_per_slice=9,
    alpha=25.0,
    quality="fast"
)
```

## Visualization

After alignment, visualize the z-stack:

```python
import numpy as np

# Load aligned volume
volume = np.load("results/aligned_volume.npy")  # (Z, H, W, C)

# For visualization, normalize and view first channel
volume_vis = volume[:, :, :, 0]  # (Z, H, W)
volume_vis = (volume_vis - volume_vis.min()) / (volume_vis.max() - volume_vis.min())

# Using napari (if available)
try:
    import napari
    viewer = napari.view_image(volume_vis, name="Aligned Z-Stack")
    napari.run()
except ImportError:
    print("Install napari for interactive 3D visualization: pip install napari[all]")
```

## Troubleshooting

**Problem**: Registration fails at deeper z-slices

**Solutions**:
- Reduce `alpha` to allow more flexible registration between different depths
- Increase `frames_per_slice` for better SNR in each slice
- Check that `update_reference=True` is set
- Verify that `buffer_size` equals `frames_per_slice`

**Problem**: Excessive motion between registered frames of the same slice

**Solutions**:
- Increase `alpha` to constrain motion within each slice
- Check acquisition timing - frames within a slice should be temporally close
- Verify `buffer_size` is set correctly (motion correction should be similar for frames within a slice)

**Problem**: Blurry or distorted final volume

**Solutions**:
- Check registration quality by visualizing displacement fields (`save_w=True`)
- Increase `quality_setting` from "fast" to "balanced" or "quality"
- Verify sufficient frames per slice (minimum 5-7 recommended)
- Check for excessive sample drift during acquisition

## Comparison with True 3D Registration

PyFlowReg's z-stack approach vs. true 3D volumetric registration:

**PyFlowReg (2D + adaptive reference)**:
- Registers each frame as a 2D image
- Adapts reference slice-by-slice
- Cannot correct through-plane (z-axis) motion
- Fast and memory-efficient
- Works well when z-motion is minimal and xy-motion dominates

**True 3D registration** (not available in PyFlowReg):
- Computes 3D displacement fields with (u, v, w) components
- Can correct motion in all three dimensions simultaneously
- Requires loading entire volume into memory
- Computationally expensive
- Needed when significant z-drift occurs during acquisition

For most 2-photon z-stack applications, PyFlowReg's 2D approach with adaptive reference updating is sufficient because:
- Z-stage motion is typically small and well-controlled
- Dominant motion artifacts occur in the xy-plane (sample drift, physiological motion)
- The adaptive reference handles gradual z-dependent anatomy changes

## See Also

- [Workflows](workflows.md) - Basic 2D time series registration workflows
- [Configuration](configuration.md) - Parameter tuning guide
- [Parameter Theory](../theory/parameters.md) - Understanding alpha and sigma parameters
