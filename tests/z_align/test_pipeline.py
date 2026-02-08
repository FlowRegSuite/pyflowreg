"""
Tests for z-align pipeline stages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyflowreg.z_align.config import ZAlignConfig
from pyflowreg.z_align import pipeline


class DummyBatchReader:
    """Simple batch reader used for stage tests."""

    def __init__(self, batches):
        self._batches = [np.asarray(b) for b in batches]
        self._idx = 0
        self.closed = False

    def has_batch(self):
        return self._idx < len(self._batches)

    def read_batch(self):
        out = self._batches[self._idx]
        self._idx += 1
        return out

    def close(self):
        self.closed = True


class RecordingWriter:
    """Writer that records frame writes and touches destination on close."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.writes = []
        self.closed = False

    def write_frames(self, frames):
        self.writes.append(np.asarray(frames))

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.closed = True


def _make_config(tmp_path: Path, **kwargs) -> ZAlignConfig:
    params = {
        "root": tmp_path,
        "input_file": "compensated.tiff",
        "output_root": "z_out",
        "volume_output_dir": "aligned_stack",
        "z_shift_file": "z_shift.HDF5",
        "corrected_output_file": "compensated_shift_corrected.tif",
        "simulated_output_file": "simulated_from_z.tif",
    }
    params.update(kwargs)
    return ZAlignConfig(**params)


class TestStatusHelpers:
    """Test status.json helper functions."""

    def test_save_and_load_status_roundtrip(self, tmp_path):
        out = tmp_path / "z_out"
        out.mkdir()
        status = {"stage1": "done", "anchor_z": 3}
        pipeline.save_status(out, status)
        loaded = pipeline.load_or_create_status(out)
        assert loaded == status

    def test_load_status_missing_returns_empty_dict(self, tmp_path):
        out = tmp_path / "no_status"
        out.mkdir()
        assert pipeline.load_or_create_status(out) == {}


class TestStage1:
    """Test stage 1 behavior."""

    def test_run_stage1_uses_existing_reference_volume(self, tmp_path):
        ref_path = tmp_path / "reference_volume.h5"
        ref_path.touch()
        cfg = _make_config(tmp_path, reference_volume="reference_volume.h5")

        result = pipeline.run_stage1(cfg)
        assert result == ref_path

        status = pipeline.load_or_create_status(cfg.resolve_output_root())
        assert status["stage1"] == "done"
        assert Path(status["volume_path"]) == ref_path

    def test_run_stage1_runs_compensate_recording(self, tmp_path, monkeypatch):
        source = tmp_path / "file_00004_00001.tif"
        source.touch()
        cfg = _make_config(tmp_path, volume_input_file="file_00004_00001.tif")

        captured = {}

        def fake_compensate_recording(options):
            captured["alpha"] = options.alpha
            captured["quality_setting"] = options.quality_setting.value
            out = Path(options.output_path)
            out.mkdir(parents=True, exist_ok=True)
            (out / "compensated.HDF5").touch()
            return None

        monkeypatch.setattr(pipeline, "compensate_recording", fake_compensate_recording)

        result = pipeline.run_stage1(cfg)

        assert result.exists()
        assert result.name.lower().endswith("hdf5")
        assert captured["alpha"] == (5.0, 5.0)
        assert captured["quality_setting"] == "quality"


class TestStage2:
    """Test stage 2 behavior."""

    def test_run_stage2_resume_skip(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path, write_corrected=False, reference_volume="vol.h5")
        volume_path = tmp_path / "vol.h5"
        volume_path.touch()

        z_shift_path = cfg.resolve_z_shift_file()
        z_shift_path.parent.mkdir(parents=True, exist_ok=True)
        z_shift_path.touch()

        pipeline.save_status(
            cfg.resolve_output_root(),
            {"stage2": "done", "anchor_z": 2, "anchor_z_1based": 3},
        )

        # If resume path is taken, _load_volume should never be called.
        monkeypatch.setattr(
            pipeline,
            "_load_volume",
            lambda *_args, **_kwargs: pytest.fail("Should not load volume on resume"),
        )

        out = pipeline.run_stage2(cfg)
        assert out["z_shift_path"] == z_shift_path
        assert out["corrected_path"] is None
        assert out["anchor_z"] == 2

    def test_run_stage2_writes_1based_zshift_and_corrected(self, tmp_path, monkeypatch):
        input_file = tmp_path / "compensated.tiff"
        input_file.touch()
        volume_file = tmp_path / "volume.h5"
        volume_file.touch()

        cfg = _make_config(
            tmp_path,
            reference_volume="volume.h5",
            write_corrected=True,
            patch_size=2,
            overlap=0.5,
            win_half=1,
        )

        H, W, C, Z = 4, 4, 1, 3
        T = 2
        batch_thwc = np.full((T, H, W, C), 100.0, dtype=np.float32)
        input_reader = DummyBatchReader([batch_thwc])

        writers = {}

        def fake_get_video_file_reader(path, *args, **kwargs):
            if Path(path) == cfg.resolve_input_file():
                return input_reader
            raise AssertionError(f"Unexpected reader path: {path}")

        def fake_get_video_file_writer(path, output_format, **kwargs):
            writer = RecordingWriter(path)
            writers[str(Path(path))] = writer
            return writer

        monkeypatch.setattr(
            pipeline, "get_video_file_reader", fake_get_video_file_reader
        )
        monkeypatch.setattr(
            pipeline, "get_video_file_writer", fake_get_video_file_writer
        )
        monkeypatch.setattr(
            pipeline,
            "_load_volume",
            lambda _cfg, _path: np.arange(H * W * C * Z, dtype=np.float32).reshape(
                H, W, C, Z
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "_compute_volume_gradients",
            lambda volume, sigma: (
                np.zeros_like(volume, dtype=np.float32),
                np.zeros_like(volume, dtype=np.float32),
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "_compute_batch_gradients",
            lambda batch, spatial_sigma, temporal_sigma: (
                np.zeros_like(batch, dtype=np.float32),
                np.zeros_like(batch, dtype=np.float32),
            ),
        )
        monkeypatch.setattr(
            pipeline,
            "_estimate_anchor_z",
            lambda gx_vol, gy_vol, gx_f, gy_f: (1, np.zeros((Z,), dtype=np.float64)),
        )
        monkeypatch.setattr(
            pipeline,
            "_estimate_z_patchwise",
            lambda *args, **kwargs: np.zeros((H, W, T), dtype=np.float64),
        )
        monkeypatch.setattr(
            pipeline,
            "_apply_z_correction",
            lambda batch_hwct, z_hat_hwt, diff_hwcz: batch_hwct + 5.0,
        )

        out = pipeline.run_stage2(cfg, volume_path=volume_file)

        z_writer = writers[str(cfg.resolve_z_shift_file())]
        corrected_writer = writers[str(cfg.resolve_corrected_output_file())]

        assert out["anchor_z"] == 1
        assert len(z_writer.writes) == 1
        assert len(corrected_writer.writes) == 1

        z_written = z_writer.writes[0]
        corrected_written = corrected_writer.writes[0]

        # z_shift is stored as 1-based IDs for MATLAB compatibility.
        assert np.all(z_written == 1.0)
        assert corrected_written.dtype == np.uint16
        assert np.all(corrected_written == 105)

        status = pipeline.load_or_create_status(cfg.resolve_output_root())
        assert status["stage2"] == "done"
        assert status["anchor_z"] == 1
        assert status["anchor_z_1based"] == 2


class TestStage3:
    """Test stage 3 behavior."""

    def test_run_stage3_subtracts_one_before_simulation(self, tmp_path, monkeypatch):
        volume_file = tmp_path / "vol.h5"
        volume_file.touch()
        cfg = _make_config(tmp_path, reference_volume="vol.h5")

        z_shift_path = cfg.resolve_z_shift_file()
        z_shift_path.parent.mkdir(parents=True, exist_ok=True)
        z_shift_path.touch()

        H, W, C, Z = 3, 3, 1, 4
        z_batch_thwc = np.full((2, H, W, 1), 2.0, dtype=np.float32)  # 1-based z=2
        z_reader = DummyBatchReader([z_batch_thwc])
        writers = {}
        seen = {}

        def fake_get_video_file_reader(path, *args, **kwargs):
            if Path(path) == z_shift_path:
                return z_reader
            raise AssertionError(f"Unexpected stage3 reader path: {path}")

        def fake_get_video_file_writer(path, output_format, **kwargs):
            writer = RecordingWriter(path)
            writers[str(Path(path))] = writer
            return writer

        def fake_simulate_from_z(volume_hwcz, z_hat_hwt):
            seen["z_hat_hwt"] = z_hat_hwt.copy()
            T = z_hat_hwt.shape[2]
            return np.full((H, W, C, T), 7.0, dtype=np.float32)

        monkeypatch.setattr(
            pipeline, "get_video_file_reader", fake_get_video_file_reader
        )
        monkeypatch.setattr(
            pipeline, "get_video_file_writer", fake_get_video_file_writer
        )
        monkeypatch.setattr(
            pipeline,
            "_load_volume",
            lambda _cfg, _path: np.zeros((H, W, C, Z), dtype=np.float32),
        )
        monkeypatch.setattr(pipeline, "_simulate_from_z", fake_simulate_from_z)

        out_path = pipeline.run_stage3(
            cfg, volume_path=volume_file, z_shift_path=z_shift_path
        )

        assert out_path == cfg.resolve_simulated_output_file()
        assert np.all(seen["z_hat_hwt"] == 1.0)  # 2 (1-based) -> 1 (0-based)

        writer = writers[str(cfg.resolve_simulated_output_file())]
        assert len(writer.writes) == 1
        assert writer.writes[0].dtype == np.uint16
        assert np.all(writer.writes[0] == 7)

        status = pipeline.load_or_create_status(cfg.resolve_output_root())
        assert status["stage3"] == "done"

    def test_run_stage3_returns_none_when_disabled(self, tmp_path):
        cfg = _make_config(tmp_path, write_simulated=False)
        assert pipeline.run_stage3(cfg) is None


class TestRunAllStages:
    """Test all-stage orchestrator behavior."""

    def test_run_all_stages_skips_stage3_when_disabled(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path, write_simulated=False)
        called = {"stage1": 0, "stage2": 0, "stage3": 0}

        monkeypatch.setattr(
            pipeline,
            "run_stage1",
            lambda config, of_options_override=None: called.__setitem__(
                "stage1", called["stage1"] + 1
            )
            or (tmp_path / "vol.h5"),
        )
        monkeypatch.setattr(
            pipeline,
            "run_stage2",
            lambda config, volume_path=None: called.__setitem__(
                "stage2", called["stage2"] + 1
            )
            or {
                "z_shift_path": tmp_path / "z_shift.HDF5",
                "corrected_path": tmp_path / "corr.tif",
                "anchor_z": 0,
            },
        )
        monkeypatch.setattr(
            pipeline,
            "run_stage3",
            lambda *args, **kwargs: called.__setitem__("stage3", called["stage3"] + 1),
        )

        out = pipeline.run_all_stages(cfg)

        assert called["stage1"] == 1
        assert called["stage2"] == 1
        assert called["stage3"] == 0
        assert out["simulated_path"] is None
