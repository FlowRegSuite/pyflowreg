# Workflows

PyFlowReg supports three primary workflows for motion correction, each optimized for different use cases.

## Array-Based Workflow

The array-based workflow is ideal for smaller datasets that fit in memory or when you need direct access to the registered data for immediate analysis.

### Basic Usage

```python
import numpy as np
from pyflowreg.motion_correction import compensate_arr, OFOptions

# Load video data (T, H, W, C)
video = np.load("my_video.npy")

# Create reference from stable frames
reference = np.mean(video[100:200], axis=0)

# Configure motion correction
options = OFOptions(quality_setting="balanced")

# Run compensation
registered, flow = compensate_arr(video, reference, options)
```

### Returns

- `registered`: Motion-corrected video with same shape as input
- `flow`: Displacement fields with shape (T, H, W, 2) containing (u, v) components

### When to Use

- Dataset fits comfortably in memory (typically <10GB)
- Need immediate access to registered frames for analysis
- Iterating on parameters and need fast feedback
- Working in Jupyter notebooks or interactive analysis

## File-Based Workflow

The file-based workflow is optimized for large datasets, providing efficient chunked processing with automatic I/O management.

### Basic Usage

```python
from pyflowreg.motion_correction import compensate_recording, OFOptions

# Configure with file paths
options = OFOptions(
    input_file="raw_video.h5",
    output_path="results/",
    output_format="HDF5",
    quality_setting="balanced",
    reference_frames=list(range(100, 200)),
    save_w=True  # Save displacement fields
)

# Run compensation (auto-selects parallelization)
compensate_recording(options)
```

### Output Files

By default, the following files are created in `output_path`:

- `<basename>_corrected.<ext>`: Registered video
- `<basename>_w.<ext>`: Displacement fields (if `save_w=True`)
- `<basename>_metadata.json`: Processing metadata and parameters

### Manual Parallelization Control

```python
from pyflowreg.motion_correction import compensate_recording, OFOptions
from pyflowreg.motion_correction.compensate_recording import RegistrationConfig

options = OFOptions(
    input_file="raw_video.h5",
    output_path="results/",
    quality_setting="balanced"
)

# Configure parallelization
config = RegistrationConfig(
    n_jobs=8,  # Use 8 cores
    batch_size=100,  # Process 100 frames per batch
    parallelization="multiprocessing"
)

compensate_recording(options, config=config)
```

### When to Use

- Large datasets that don't fit in memory
- Production processing pipelines
- Want to save results to disk for later analysis
- Need parallelization for faster processing

## Real-Time Processing

The real-time workflow enables online motion correction with adaptive reference updating, suitable for live imaging or streaming analysis.

### Basic Usage

```python
import numpy as np
from pyflowreg.motion_correction import FlowRegLive, OFOptions

# Configure for speed
options = OFOptions(quality_setting="fast")

# Initialize processor
processor = FlowRegLive(options)

# Initialize reference from first N frames
initial_frames = np.stack([camera.grab() for _ in range(10)])
processor.set_reference(initial_frames)

# Process streaming frames
while imaging:
    frame = camera.grab()
    corrected, flow = processor(frame)
    display(corrected)
```

### Adaptive Reference

The reference can be updated during processing to adapt to slow changes:

```python
# Update reference every 20 frames with 20% weight
frame_count = 0
for frame in video_stream:
    corrected, flow = processor(frame)

    frame_count += 1
    if frame_count % 20 == 0:
        processor.update_reference(corrected, weight=0.2)
```

### When to Use

- Live microscopy with real-time correction
- Streaming data from cameras
- Interactive visualization during acquisition
- Fast preview of motion artifacts

## Choosing a Workflow

| Criterion | Array-Based | File-Based | Real-Time |
|-----------|-------------|------------|-----------|
| Dataset size | Small (<10GB) | Large (>10GB) | Streaming |
| Memory usage | High | Low | Low |
| Speed | Fast | Moderate | Fastest |
| Quality | Best | Best | Good |
| Parallelization | No | Yes | No |
| Output | In-memory | Files | In-memory |
| Use case | Analysis | Production | Live imaging |
