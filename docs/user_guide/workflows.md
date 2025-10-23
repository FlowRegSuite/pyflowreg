# Workflows

PyFlowReg supports three primary workflows for motion correction, each optimized for different use cases. All workflows support flexible output handling and real-time data access through callbacks.

## Output Formats

All workflows support configurable output strategies through the `output_format` parameter:

### Standard File Formats
- `OutputFormat.HDF5` - HDF5 file storage
- `OutputFormat.TIFF` - TIFF stack output
- `OutputFormat.MAT` - MATLAB compatible files

### Memory Formats
- `OutputFormat.ARRAY` - Accumulate in memory, return as array
- `OutputFormat.NULL` - Discard output (callback-only processing)

### Example: Choosing Output Strategy

```python
from pyflowreg.motion_correction import compensate_arr
from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat
import numpy as np

video = np.random.rand(100, 256, 256, 2)
reference = np.mean(video[:10], axis=0)

# Array-based workflow (compensate_arr) always returns arrays
# For callback-only processing, use compensate_recording with NULL writer

# File storage
options_file = OFOptions(
    output_format=OutputFormat.HDF5,
    output_path="results/"
)
```

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

## Real-Time Data Access via Callbacks

The API provides callbacks for accessing data during processing, enabling real-time visualization, monitoring, and analysis without waiting for completion. This works with all workflows.

### Available Callbacks

| Callback | Data Provided | Use Cases |
|----------|---------------|-----------|
| `progress_callback` | `(current_frame, total_frames)` | Progress bars, status updates |
| `w_callback` | `(w_batch, start_idx, end_idx)` | Motion tracking, flow analysis |
| `registered_callback` | `(batch, start_idx, end_idx)` | Live display, quality metrics |

### Basic Callback Usage

```python
def monitor_motion(w_batch, start_idx, end_idx):
    """Monitor displacement fields during processing."""
    for t in range(w_batch.shape[0]):
        magnitude = np.sqrt(
            w_batch[t, :, :, 0]**2 + w_batch[t, :, :, 1]**2
        )
        print(f"Frame {start_idx + t}: mean motion = {np.mean(magnitude):.2f}")

def display_frames(batch, start_idx, end_idx):
    """Display corrected frames as they're computed."""
    # Update your display/viewer with the batch
    pass

# Use callbacks during processing
registered, w = compensate_arr(
    video,
    reference,
    options=options,
    w_callback=monitor_motion,
    registered_callback=display_frames
)
```

### Using Callbacks

Process data during registration without explicit storage using callbacks:

```python
class LiveProcessor:
    def __init__(self):
        self.statistics = []

    def process_batch(self, w_batch, start_idx, end_idx):
        # Process displacement fields during registration
        batch_stats = {
            'start': start_idx,
            'end': end_idx,
            'mean_motion': np.mean(np.sqrt(
                w_batch[..., 0]**2 + w_batch[..., 1]**2
            ))
        }
        self.statistics.append(batch_stats)

processor = LiveProcessor()

# compensate_arr returns arrays but also calls callbacks
registered, w = compensate_arr(
    video,
    reference,
    options=OFOptions(quality_setting="balanced"),
    w_callback=processor.process_batch
)

# Access collected statistics
print(f"Processed {len(processor.statistics)} batches")
```

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
    quality_setting="balanced",
    buffer_size=100,  # Process 100 frames per batch
)

# Configure parallelization
config = RegistrationConfig(
    n_jobs=8,  # Use 8 cores
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

The reference can be reset or updated during processing:

```python
# Set new reference from array of frames
reference_frames = np.stack([corrected_frames[:10]])
processor.set_reference(reference_frames)

# Or reset reference buffer
processor.reset_reference()
```

### When to Use

- Live microscopy with real-time correction
- Streaming data from cameras
- Interactive visualization during acquisition
- Fast preview of motion artifacts

## Integration with Visualization Tools

The API is designed for seamless integration with visualization tools like napari:

```python
import napari
from pyflowreg.motion_correction import compensate_arr
from pyflowreg.motion_correction.OF_options import OFOptions, OutputFormat

class NapariLiveCorrection:
    """Live motion correction display in napari."""

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.corrected_layer = None

    def setup(self, shape):
        """Initialize display layers."""
        T, H, W, C = shape
        # Pre-allocate corrected video layer
        self.corrected_layer = self.viewer.add_image(
            np.zeros((T, H, W, C), dtype=np.float32),
            name='Motion Corrected',
            colormap='viridis'
        )

    def update_display(self, batch, start_idx, end_idx):
        """Update napari display with new batch."""
        # Update the corrected layer data
        self.corrected_layer.data[start_idx:end_idx] = batch

        # Move viewer to show latest frame
        self.viewer.dims.set_current_step(0, end_idx - 1)

    def correct_with_display(self, video, reference):
        """Run correction with live display."""
        self.setup(video.shape)

        options = OFOptions(
            buffer_size=10,
            levels=5,
            iterations=50
        )

        # compensate_arr always returns arrays
        return compensate_arr(
            video,
            reference,
            options=options,
            registered_callback=self.update_display
        )

# Usage
viewer = napari.Viewer()
corrector = NapariLiveCorrection(viewer)

video = load_video()  # Your video loading
reference = compute_reference(video)

corrected, w = corrector.correct_with_display(video, reference)
napari.run()
```

## Performance Optimization

### Buffer Size Selection

Buffer size affects callback frequency and memory usage:

```python
# Real-time display - smaller buffers for responsive updates
options_realtime = OFOptions(
    buffer_size=10,  # Update every 10 frames
    output_format=OutputFormat.NULL
)

# Batch processing - larger buffers for efficiency
options_batch = OFOptions(
    buffer_size=100,  # Process 100 frames at once
    output_format=OutputFormat.HDF5
)
```

### Callback Performance Guidelines

1. **Keep callbacks fast** - They run in the processing thread
2. **Copy data if storing** - Don't rely on references to batch data
3. **Use threading for heavy operations** - Offload visualization/IO

```python
import threading
from queue import Queue

class AsyncProcessor:
    def __init__(self):
        self.queue = Queue()
        self.worker = threading.Thread(target=self._process_queue)
        self.worker.start()

    def callback(self, batch, start_idx, end_idx):
        # Quick copy and queue
        self.queue.put((batch.copy(), start_idx, end_idx))

    def _process_queue(self):
        while True:
            batch, start, end = self.queue.get()
            # Heavy processing here
            self._heavy_processing(batch)
```

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

## API Reference Summary

### Callback Signatures

```python
# Progress callback
def progress(current: int, total: int) -> None:
    """Called with current frame number and total frames."""
    pass

# Displacement field callback
def w_callback(w_batch: np.ndarray, start_idx: int, end_idx: int) -> None:
    """Called with batch of displacement fields.

    Args:
        w_batch: Shape (T, H, W, 2) with [u, v] components
        start_idx: Starting frame index in overall video
        end_idx: Ending frame index (exclusive)
    """
    pass

# Registered frame callback
def registered_callback(batch: np.ndarray, start_idx: int, end_idx: int) -> None:
    """Called with batch of corrected frames.

    Args:
        batch: Shape (T, H, W, C) corrected frames
        start_idx: Starting frame index in overall video
        end_idx: Ending frame index (exclusive)
    """
    pass
```

### Key Configuration Options

```python
OFOptions(
    output_format=OutputFormat.NULL,  # NULL, ARRAY, HDF5, TIFF, etc.
    buffer_size=20,                   # Frames per batch
    save_w=True,                       # Compute displacement fields
    levels=5,                          # Pyramid levels
    iterations=50,                     # Iterations per level
    quality_setting="balanced",       # fast, balanced, accurate
)
```
