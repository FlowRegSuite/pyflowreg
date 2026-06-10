# Examples Gallery

PyFlowReg ships runnable example scripts in `examples/` and Jupyter notebooks in `notebooks/`. This page describes what each tracked example demonstrates, the data it needs, how to run it, and what output to expect.

## Running the examples

Always run example scripts as modules from the project root:

```bash
python -m examples.jupiter_demo
```

Do not `cd` into `examples/` or add `sys.path` modifications.

Most examples download their demo data automatically on first run via `pyflowreg.util.download.download_demo_data()`. The file is cached in the `data/` folder at the project root, so subsequent runs reuse the local copy. Available demo files are `jupiter.tiff`, `synth_frames.h5`, and `injection.tiff`. The exception is `z_shift_demo.py`, which requires user-supplied input files (see below).

Several demos open an OpenCV playback window. The common keyboard controls are `q` (quit), `p` (pause/resume), and `r` (restart).

## Example scripts

### jupiter_demo.py — file-based motion correction

Demonstrates the file-based workflow with `compensate_recording()` and a minimal `OFOptions` configuration: `alpha=4`, `quality_setting="balanced"`, HDF5 output, and frames 100-200 averaged as the reference (`reference_frames=list(range(100, 201))`).

- **Data:** downloads `jupiter.tiff` automatically.
- **Run:** `python -m examples.jupiter_demo`
- **Output:** writes the compensated recording to `jupiter_demo/hdf5_comp_minimal/compensated.HDF5` (relative to the working directory), then reads both the original and compensated videos back with `get_video_file_reader()` and opens an OpenCV window ("Jupiter Demo - Comparison") with side-by-side playback of uncorrected vs. corrected frames.

### jupiter_demo_arr.py — in-memory array workflow

Same data and parameters as `jupiter_demo.py`, but processes the video entirely in memory with `compensate_arr()`: the whole recording is read into an array, the reference is computed as the mean of frames 100-200, and `compensate_arr(video_array, reference, options)` returns the registered video `(T, H, W, C)` and the displacement fields `(T, H, W, 2)` with components (u, v) — u horizontal (x), v vertical (y). The options set `save_w=True` and `output_typename="double"`. See [Motion Correction API](../api/motion_correction.md) for how `compensate_arr` handles output (it returns arrays rather than writing files).

- **Data:** downloads `jupiter.tiff` automatically.
- **Run:** `python -m examples.jupiter_demo_arr`
- **Output:** no files are written. Prints displacement statistics (max/mean magnitude) and opens an OpenCV window ("Jupiter Demo Array - Comparison") with side-by-side playback.

### jupiter_demo_arr_gpu.py — GPU flow backend

Identical to `jupiter_demo_arr.py` except the flow backend is switched to PyTorch via `flow_backend="flowreg_torch"` with `backend_params={"device": "cuda", "dtype": "float32"}`, and the processing time is measured. The script docstring notes that the CuPy backend (`flowreg_cuda`) is also available on systems with CUDA. See [Flow Backends](backends.md) for backend selection and [Parallelization](parallelization.md) for how GPU backends interact with executors.

- **Data:** downloads `jupiter.tiff` automatically.
- **Run:** `python -m examples.jupiter_demo_arr_gpu` (requires PyTorch; the script requests the `cuda` device)
- **Output:** no files are written. Prints elapsed time and displacement statistics, then opens an OpenCV window ("Jupiter Demo GPU - Comparison") with side-by-side playback.

### jupiter_demo_live.py — online motion correction with FlowRegLive

Demonstrates frame-by-frame online processing with the `FlowRegLive` class (see [Online Processing](online_processing.md)). The reference is set from frames 100-200 via `set_reference()`, and the instance is configured with `reference_buffer_size=50`, `reference_update_interval=20`, and `reference_update_weight=0.2`, so a corrected frame is mixed into the reference every 20 frames with weight 0.2. Each loop iteration calls `flow_reg(frame)`, which returns the corrected frame `(H, W, C)` and displacement field `(H, W, 2)`.

- **Data:** downloads `jupiter.tiff` automatically.
- **Run:** `python -m examples.jupiter_demo_live`
- **Output:** no files are written. Opens an OpenCV window ("Jupiter Live Demo - Real-time Compensation") that loops endlessly over the video, showing original vs. corrected frames with an FPS counter, per-frame processing time, flow-magnitude readout, and a "REF UPDATE" indicator on reference-update frames.

### injection_session_demo.py — multi-recording session pipeline

Demonstrates the three-stage session pipeline (see [Multi-Session Processing](multi_session.md)): Stage 1 per-recording motion correction, Stage 2 inter-sequence alignment, Stage 3 valid-mask computation. The script first builds a synthetic multi-file dataset: it downloads `injection.tiff`, creates three spatially shifted and cropped variants, writes them as `injection_0.tif` to `injection_2.tif` with `get_video_file_writer()`, then configures a `SessionConfig` and calls `run_all_stages()`.

- **Data:** downloads `injection.tiff` automatically.
- **Run:** `python -m examples.injection_session_demo`
- **Output:** creates `injection_session_demo/` (relative to the working directory) containing the three input variants, per-recording results in `injection_session_demo/compensated_outputs/`, and session-wide results in `injection_session_demo/final_results/`, including `final_valid_idx.png` with the session-wide valid mask. Console output only; no display windows.

### synth_evaluation.py — accuracy evaluation on synthetic ground truth

Replicates the synthetic-data evaluation from the Flow-Registration paper. It loads frame pairs (clean, 35 dB, and 30 dB noise levels) and the ground-truth displacement field from `synth_frames.h5` using `h5py` (this file stores evaluation arrays, not a video), computes flow with `pyflowreg.get_displacement()` using explicit solver parameters (`alpha=(8, 8)`, `iterations=100`, `levels=50`, `eta=0.8`, ...), and reports the endpoint error (EPE) over a border-cropped region together with per-call timing. Each dataset is evaluated with `min_level=0` (refinement down to level 0, the finest, full-resolution pyramid level) and `min_level=5` (stopping at a coarser level, labeled "fast"), both for each channel individually and for both channels combined.

- **Data:** downloads `synth_frames.h5` automatically.
- **Run:** `python -m examples.synth_evaluation` (the script body runs under `if __name__ == "__main__":`)
- **Output:** console output only (timings and EPE values); no files or windows.

### z_shift_demo.py — z-alignment workflow

Demonstrates the MATLAB-style z-shift correction workflow using `pyflowreg.z_align` (`ZAlignConfig` and `run_all_stages`); see [Z-Alignment](z_align.md) and the [z_align API](../api/z_align.md). The configuration builds a reference volume from a stack file, estimates per-patch z shifts for a time recording, and writes a z-corrected signal and a z-shift-only simulated video.

- **Data:** not downloaded. Requires two user-supplied files in the working directory: `compensated.tiff` (the time recording to z-correct) and `file_00004_00001.tif` (the stack used to build the reference volume). The script raises `FileNotFoundError` if either is missing.
- **Run:** `python -m examples.z_shift_demo` from the directory containing the input files (when run from the project root, place the inputs there).
- **Output:** writes `aligned_stack/compensated.HDF5` (reference volume), `z_shift.HDF5`, `compensated_shift_corrected.tif`, and `simulated_from_z.tif` to the working directory, matching the MATLAB output names.

## Configuration templates

Three tracked files in `examples/` are configuration templates rather than scripts:

- `examples/session_config.toml` and `examples/session_config.yml` — annotated templates for `SessionConfig`, used with the session CLI: `pyflowreg-session run --config session_config.toml`. See [Multi-Session Processing](multi_session.md).
- `examples/z_align_config.toml` — annotated template for `ZAlignConfig` matching `z_shift_demo.py`, used with the z-align CLI: `pyflowreg-z-align run --config z_align_config.toml`. See [Z-Alignment](z_align.md).

## Notebooks

Notebook-specific dependencies are listed in `notebooks/requirements_notebooks.txt` (matplotlib, IPython).

### jupiter_demo.ipynb

Launch with `jupyter notebook notebooks/jupiter_demo.ipynb`.

An extended version of the Jupiter demo with quantitative analysis. The notebook:

1. Downloads `jupiter.tiff` (cached in `data/`).
2. Simulates a two-channel recording by reading the single-channel TIFF twice and converts it to HDF5 in two layouts (a single file with `ch1`/`ch2` datasets and a multi-file variant) using `get_video_file_reader()`/`get_video_file_writer()` batch I/O.
3. Runs `compensate_recording()` with `alpha=4`, `min_level=3`, and frames 100-200 as reference, skipping the step if a compensated file already exists.
4. Compares original and compensated videos: average frames, standard-deviation maps as a motion-blur indicator, temporal slices through an impact location, time courses of relative intensity change, frame-by-frame comparisons, and an inline side-by-side animation.
5. Plots motion statistics (mean/max displacement, divergence, translation) from `statistics.npz` if `save_meta_info` produced it.

Outputs are written under `jupiter_demo/` relative to the notebook's working directory.

### flow_visualization.ipynb

Launch with `jupyter notebook notebooks/flow_visualization.ipynb`.

Demonstrates the flow visualization utilities in `pyflowreg.util.visualization` on synthetic data with known ground truth. The notebook:

1. Downloads `synth_frames.h5` and loads clean frames plus the ground-truth displacement field (reordering the stored components to match the PyFlowReg (u, v) convention).
2. Computes flow with `pyflowreg.get_displacement()` using the same parameters as `synth_evaluation.py` and reports the endpoint error against ground truth. The frames and ground-truth flow are rotated 90 degrees as a worked example of correctly rotating vector fields.
3. Shows Middlebury color coding with `flow_to_color()` (hue encodes direction, saturation encodes magnitude) including a color-wheel legend, quiver plots with and without streamlines via `quiver_visualization()` (matplotlib backend; an OpenCV backend is also available), two-channel color mapping with `get_visualization()`, and an error-magnitude map, ending with a combined overview figure.
