"""
Smoke tests for the demo scripts in examples/.

Tier 1 (this module, always on): every tracked example script imports
without side effects and exposes the expected entry point. The scripts
live outside the package (examples/ has no __init__.py), so they are
loaded from their file paths.

Tier 2/3 (full main() runs on synthetic/cached demo data) live in
classes marked ``slow`` / ``demo_data`` further down.
"""

import importlib.util
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pyflowreg.util.io.factory import get_video_file_reader, get_video_file_writer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

EXAMPLE_SCRIPTS = [
    "injection_session_demo",
    "jupiter_demo",
    "jupiter_demo_arr",
    "jupiter_demo_arr_gpu",
    "jupiter_demo_live",
    "synth_evaluation",
    "z_shift_demo",
]

SCRIPTS_WITH_MAIN = list(EXAMPLE_SCRIPTS)


def _import_example(script_name):
    path = EXAMPLES_DIR / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"_example_{script_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExampleImportability:
    """Tier 1: importing an example must be cheap and side-effect free."""

    @pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
    def test_example_import_has_no_side_effects(
        self, script_name, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _import_example(script_name)
        created = list(tmp_path.iterdir())
        assert created == [], (
            f"Importing examples/{script_name}.py created files: {created}. "
            "Example scripts must do all work inside main()/__main__."
        )

    @pytest.mark.parametrize("script_name", SCRIPTS_WITH_MAIN)
    def test_example_exposes_main(self, script_name, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        module = _import_example(script_name)
        assert callable(
            getattr(module, "main", None)
        ), f"examples/{script_name}.py should expose a callable main()"


# ---------------------------------------------------------------------------
# Tier 2/3 infrastructure: synthetic demo inputs and headless cv2
# ---------------------------------------------------------------------------

# The jupiter demos build their reference from frames 100:201, so the
# synthetic stand-in for jupiter.tiff needs at least 201 frames; spatial
# size is kept tiny so the variational solver stays fast.
JUPITER_SHAPE = (210, 32, 48, 1)

# injection_session_demo shifts the video by +/-50 px and crops a 50 px
# border, so H and W must exceed 100; 128x128 leaves 28x28 variants.
# The demo rescales with ``256 * video.astype(np.uint16)``, which expects
# uint8 input (like the real injection.tiff).
INJECTION_SHAPE = (10, 128, 128, 1)


def _write_video(path, frames, output_format="TIFF"):
    """Write synthetic frames through pyflowreg's own writer."""
    writer = get_video_file_writer(str(path), output_format)
    try:
        writer.write_frames(frames)
    finally:
        writer.close()


def _moving_disc_video(shape, seed=0):
    """(T, H, W, 1) uint16 video of a bright disc on a circular path.

    Gives the flow solver real structure to track (same pattern the docs
    test harness uses for synthetic inputs).
    """
    T, H, W, C = shape
    assert C == 1, "helper generates single-channel videos"
    rng = np.random.default_rng(seed)
    frames = np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[:H, :W]
    radius = max(2, min(H, W) // 8)
    for t in range(T):
        off_y = int(round(radius * 0.5 * np.sin(2 * np.pi * t / T)))
        off_x = int(round(radius * 0.5 * np.cos(2 * np.pi * t / T)))
        mask = (x - W // 2 - off_x) ** 2 + (y - H // 2 - off_y) ** 2 <= radius**2
        frame = mask.astype(np.float32) * 0.8 + 0.2 * rng.random(
            (H, W), dtype=np.float32
        )
        frames[t] = frame[:, :, None]
    return (np.clip(frames, 0.0, 1.0) * 65535).astype(np.uint16)


def _tiled_texture_video(shape, seed=0):
    """(T, H, W, 1) uint8 video with bright squares tiled everywhere.

    injection_session_demo crops three disjoint 28x28 regions out of the
    frame, so every region must contain structure for the session pipeline
    (xcorr pre-alignment + flow refinement) to have something to lock onto.
    """
    T, H, W, C = shape
    assert C == 1, "helper generates single-channel videos"
    rng = np.random.default_rng(seed)
    base = rng.random((H, W), dtype=np.float32) * 0.3
    for y0 in range(4, H - 8, 16):
        for x0 in range(4, W - 8, 16):
            base[y0 : y0 + 6, x0 : x0 + 6] += 0.5
    frames = np.zeros(shape, dtype=np.float32)
    for t in range(T):
        noisy = base + 0.05 * rng.random((H, W), dtype=np.float32)
        frames[t] = np.clip(noisy, 0.0, 1.0)[:, :, None]
    return (frames * 255).astype(np.uint8)


def _write_synth_frames_h5(path, size=64, seed=0):
    """Create synth_frames.h5 exactly as examples/synth_evaluation.py reads it.

    The script opens the file with h5py directly (not via pyflowreg I/O)
    and expects:

    - ``clean``, ``noisy35db``, ``noisy30db``: (2, C, H, W) frame pairs with
      C=2 channels (the script's ``weight`` has two entries),
    - ``w``: (2, H, W) ground-truth flow.

    ``epe()`` crops a 25-pixel border, so H and W must exceed 50.
    """
    rng = np.random.default_rng(seed)
    # Smooth random texture per channel so flow estimation is well-posed.
    frame1 = gaussian_filter(
        rng.random((2, size, size)).astype(np.float32), sigma=(0, 3, 3)
    )
    # Second frame: 1-pixel shift along x (matching ground truth below).
    frame2 = np.roll(frame1, shift=1, axis=2)
    clean = np.stack([frame1, frame2], axis=0)  # (2, C, H, W)
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    w = np.zeros((2, size, size), dtype=np.float32)
    w[1] = 1.0
    with h5py.File(path, "w") as f:
        f.create_dataset("clean", data=clean)
        f.create_dataset("noisy35db", data=clean + 0.01 * noise)
        f.create_dataset("noisy30db", data=clean + 0.03 * noise)
        f.create_dataset("w", data=w)


def _patch_download(monkeypatch, module, path):
    """Replace the example module's download_demo_data with a stub."""
    monkeypatch.setattr(module, "download_demo_data", lambda *args, **kwargs: path)


@pytest.fixture
def headless_cv2(monkeypatch):
    """Disable cv2 GUI calls and make every display loop exit immediately.

    All jupiter demos break their playback loop when ``waitKey`` returns
    ``ord('q')`` (after ``& 0xFF``, which is a no-op for 'q'), so patching
    waitKey ends each loop after a single displayed frame.
    """
    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args, **kwargs: ord("q"))


# jupiter_demo_arr and jupiter_demo_arr_gpu wrap their pipeline in a
# try/except that only prints the traceback, so a bare main() call would
# pass even on failure; the tests therefore assert on the printed success
# marker and on the absence of the error marker.
ARRAY_DEMO_CASES = [
    pytest.param(
        "jupiter_demo_arr",
        "Motion compensation complete!",
        "Error during array compensation",
        id="jupiter_demo_arr",
    ),
    pytest.param(
        "jupiter_demo_arr_gpu",
        "Motion compensation complete!",
        "Error during GPU array compensation",
        id="jupiter_demo_arr_gpu",
    ),
    pytest.param(
        "jupiter_demo_live",
        "Live demo finished.",
        None,
        id="jupiter_demo_live",
    ),
]


@pytest.mark.slow
class TestExampleMainTinyInput:
    """Tier 2: run each example's main() end-to-end on tiny synthetic data.

    download_demo_data is monkeypatched on the example module to return a
    synthetic file with the structure the script expects, cv2 display is
    headless, and the working directory is an isolated tmp_path.

    z_shift_demo has no Tier 2 test: it requires pre-existing files from a
    MATLAB Flow-Registration workflow (compensated.tiff and
    file_00004_00001.tif) in the working directory and raises
    FileNotFoundError otherwise, so there is no synthetic-input entry point
    to exercise without reproducing that whole workflow.
    """

    def test_jupiter_demo_main_writes_compensated_video(
        self, tmp_path, monkeypatch, headless_cv2
    ):
        monkeypatch.chdir(tmp_path)
        module = _import_example("jupiter_demo")

        input_path = tmp_path / "jupiter.tiff"
        _write_video(input_path, _moving_disc_video(JUPITER_SHAPE))
        _patch_download(monkeypatch, module, input_path)

        module.main()

        compensated = (
            tmp_path / "jupiter_demo" / "hdf5_comp_minimal" / "compensated.HDF5"
        )
        assert (
            compensated.exists()
        ), "compensate_recording should write compensated.HDF5"

        reader = get_video_file_reader(str(compensated))
        frames = reader[:]
        reader.close()
        assert frames.shape[0] == JUPITER_SHAPE[0]
        assert frames.shape[1:3] == JUPITER_SHAPE[1:3]

    @pytest.mark.parametrize("script_name,success_text,error_text", ARRAY_DEMO_CASES)
    def test_array_demo_main_completes_headless(
        self,
        script_name,
        success_text,
        error_text,
        tmp_path,
        monkeypatch,
        headless_cv2,
        capsys,
    ):
        if script_name == "jupiter_demo_arr_gpu":
            pytest.importorskip("torch")
            # The script does NOT hard-require CUDA: the flowreg_torch
            # backend factory falls back to CPU with a warning when
            # device="cuda" is requested but unavailable, so no CUDA skip.

        monkeypatch.chdir(tmp_path)
        module = _import_example(script_name)

        input_path = tmp_path / "jupiter.tiff"
        _write_video(input_path, _moving_disc_video(JUPITER_SHAPE))
        _patch_download(monkeypatch, module, input_path)

        module.main()

        out = capsys.readouterr().out
        assert success_text in out
        if error_text is not None:
            assert error_text not in out

    def test_synth_evaluation_main_prints_epe_report(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        module = _import_example("synth_evaluation")

        h5_path = tmp_path / "synth_frames.h5"
        _write_synth_frames_h5(h5_path)
        _patch_download(monkeypatch, module, h5_path)

        module.main()

        out = capsys.readouterr().out
        # 3 datasets x (2 multi-channel runs + 2 channels x 2 runs) = 18
        assert out.count("Elapsed time") == 18
        for name in ("clean", "noise35db", "noise30db"):
            assert f"for {name}, default, ch 1 + 2" in out
            assert f"for {name}, fast, ch 1 + 2" in out
            assert f"for {name}, default, ch 1" in out
            assert f"for {name}, fast, ch 2" in out

    def test_injection_session_demo_main_runs_session_pipeline(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        module = _import_example("injection_session_demo")

        input_path = tmp_path / "injection.tiff"
        _write_video(input_path, _tiled_texture_video(INJECTION_SHAPE))
        _patch_download(monkeypatch, module, input_path)

        module.main()

        demo_folder = tmp_path / "injection_session_demo"

        # Stage 0: the three shifted variants were written.
        for idx in range(3):
            assert (demo_folder / f"injection_{idx}.tif").exists()

        # Stage 1: per-recording compensation outputs.
        for idx in range(3):
            out_dir = demo_folder / "compensated_outputs" / f"injection_{idx}"
            assert any(
                (out_dir / name).exists()
                for name in ("compensated.HDF5", "compensated.hdf5")
            ), f"missing compensated video for injection_{idx}"
            assert (out_dir / "temporal_average.npy").exists()

        # Stage 3: session-wide valid mask.
        final_results = demo_folder / "final_results"
        assert (final_results / "final_valid_idx.png").exists()
        assert (final_results / "final_valid_idx.npz").exists()

        final = np.load(str(final_results / "final_valid_idx.npz"))
        crop = INJECTION_SHAPE[1] - 100  # H - 2*crop_border
        assert final["final_valid"].shape == (crop, crop)


# Tier 3: same main() runs, but on the real cached demo recordings. The
# session-scoped demo_data_file fixture (tests/conftest.py) resolves files
# cached under data/ (downloading only when PYFLOWREG_TEST_DEMO_DATA=1)
# and skips the test otherwise.
DEMO_DATA_CASES = [
    pytest.param("jupiter_demo", "jupiter.tiff", None, id="jupiter_demo"),
    pytest.param(
        "jupiter_demo_arr",
        "jupiter.tiff",
        "Error during array compensation",
        id="jupiter_demo_arr",
    ),
    pytest.param(
        "jupiter_demo_arr_gpu",
        "jupiter.tiff",
        "Error during GPU array compensation",
        id="jupiter_demo_arr_gpu",
    ),
    pytest.param("jupiter_demo_live", "jupiter.tiff", None, id="jupiter_demo_live"),
    pytest.param("synth_evaluation", "synth_frames.h5", None, id="synth_evaluation"),
    pytest.param(
        "injection_session_demo", "injection.tiff", None, id="injection_session_demo"
    ),
]


@pytest.mark.demo_data
@pytest.mark.slow
class TestExampleMainDemoData:
    """Tier 3: full main() runs against the real demo recordings."""

    @pytest.mark.parametrize("script_name,demo_name,error_text", DEMO_DATA_CASES)
    def test_example_main_runs_on_demo_data(
        self,
        script_name,
        demo_name,
        error_text,
        demo_data_file,
        tmp_path,
        monkeypatch,
        headless_cv2,
        capsys,
    ):
        if script_name == "jupiter_demo_arr_gpu":
            pytest.importorskip("torch")

        data_path = demo_data_file(demo_name)

        monkeypatch.chdir(tmp_path)
        module = _import_example(script_name)
        _patch_download(monkeypatch, module, data_path)

        module.main()

        if error_text is not None:
            # These scripts swallow exceptions in a try/except that prints
            # the traceback, so check the captured output explicitly.
            assert error_text not in capsys.readouterr().out
