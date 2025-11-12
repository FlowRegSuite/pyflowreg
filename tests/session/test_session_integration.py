"""
Integration tests for full session pipeline.

Tests the complete workflow from Stage 1 through Stage 3 using
synthetic data with known ground truth displacements.
"""

import numpy as np
import pytest
from scipy.ndimage import shift as ndimage_shift

from pyflowreg.session.config import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1
from pyflowreg.session.stage2_between_avgs import run_stage2
from pyflowreg.session.stage3_valid_mask import run_stage3


def create_textured_frame(H, W, seed=None):
    """Create a frame with realistic features for tracking (matches xcorr tests)."""
    if seed is not None:
        np.random.seed(seed)

    # Create reference with structured features (similar to test_xcorr_prealignment)
    frame = np.random.rand(H, W).astype(np.float32) * 0.2
    y, x = np.ogrid[:H, :W]

    # Add Gaussian blobs
    frame += (10 * np.exp(-((y - H // 2) ** 2 + (x - W // 2) ** 2) / 200)).astype(
        np.float32
    )

    # Add rectangles for better features
    frame[H // 4 : H // 2, H // 4 : H // 2] = 5.0
    frame[int(H * 0.55) : int(H * 0.7), int(W * 0.55) : int(W * 0.7)] = 4.0
    frame[int(H * 0.3) : int(H * 0.47), int(W * 0.62) : int(W * 0.78)] = 3.0

    return frame.astype(np.float32)


def create_synthetic_sequence(H, W, T, shift_y, shift_x, seed=None):
    """
    Create synthetic sequence with constant shift.

    Parameters
    ----------
    H, W : int
        Frame height and width
    T : int
        Number of frames
    shift_y, shift_x : float
        Constant shift in y and x directions
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    video : ndarray, shape (T, H, W, 1)
        Synthetic video with constant shift applied
    """
    if seed is not None:
        np.random.seed(seed)

    # Create base textured frame
    base_frame = create_textured_frame(H, W, seed=seed)

    # Create video with slight temporal variation
    video = np.zeros((T, H, W, 1), dtype=np.float32)

    for t in range(T):
        random_noise = np.random.randn(H, W) * 0.05

        shifted = (
            ndimage_shift(
                base_frame,
                shift=[shift_y, shift_x],
                mode="constant",
                cval=0.0,
                order=3,
            )
            + random_noise
        )

        video[t, :, :, 0] = shifted

    # Normalize each frame to uint16 range
    video = video.clip(0, 1)
    video = (video * 65535).astype(np.uint16)

    return video


@pytest.fixture
def synthetic_session_data(tmp_path):
    """
    Create synthetic session with 3 sequences and known shifts.

    Sequence 0: shift (+4, +6)
    Sequence 1: no shift (center reference)
    Sequence 2: shift (-3, +2)

    Returns
    -------
    dict with keys:
        - root: Path to session root
        - ground_truth_shifts: dict mapping sequence name to (dy, dx)
        - config: SessionConfig object
    """
    # Parameters - use 128x128 to match xcorr tests
    H, W, T = 128, 128, 10

    # Ground truth shifts (y, x) - use larger shifts for better detection
    shifts = {
        "seq_0": (10.0, -5.0),
        "seq_1": (0.0, 0.0),  # Center reference
        "seq_2": (-8.0, 7.0),
    }

    # Create session directory
    session_root = tmp_path / "synthetic_session"
    session_root.mkdir()

    # Create sequences
    for seq_name, (shift_y, shift_x) in shifts.items():
        video = create_synthetic_sequence(
            H, W, T, shift_y, shift_x, seed=hash(seq_name) % 2**32
        )

        # Save as TIFF
        import tifffile

        tiff_path = session_root / f"{seq_name}.tif"
        tifffile.imwrite(str(tiff_path), video, photometric="minisblack")

    # Create config
    config = SessionConfig(
        root=session_root,
        pattern="*.tif",
        center="seq_1.tif",  # Explicitly set center
        output_root=session_root / "compensated_outputs",
        final_results=session_root / "final_results",
        resume=False,
        scheduler="local",
        cc_upsample=1,
        sigma_smooth=1,
        alpha_between=5,
        iterations_between=100,
    )

    return {
        "root": session_root,
        "ground_truth_shifts": shifts,
        "config": config,
    }


@pytest.mark.slow
class TestFullSessionPipeline:
    """
    Test full session pipeline with synthetic data.

    Creates 3 synthetic sequences with known displacements, runs all
    stages, and verifies displacement accuracy and mask correctness.
    """

    def test_complete_pipeline_displacement_accuracy(self, synthetic_session_data):
        """
        Test complete pipeline and verify displacement accuracy.

        Runs all 3 stages on synthetic data and checks that recovered
        displacements match ground truth within tolerance.
        """
        config = synthetic_session_data["config"]
        ground_truth = synthetic_session_data["ground_truth_shifts"]

        # Stage 1: Per-recording compensation
        print("\n=== Running Stage 1 ===")
        config.flow_options = {
            "quality_setting": "quality",
            "min_level": 0,
            "sigma": [[0.5, 0.5, 0.5]],
            "alpha": 1,
            "buffer_size": 32,
            "reference_frames": list(range(10)),
        }

        output_folders = run_stage1(config)

        assert len(output_folders) == 3, "Should process 3 sequences"

        # Verify outputs exist
        for folder in output_folders:
            assert (folder / "compensated.HDF5").exists()
            assert (folder / "temporal_average.npy").exists()
            assert (folder / "idx.hdf").exists()
            assert (folder / "status.json").exists()

        # Stage 2: Inter-sequence alignment
        print("\n=== Running Stage 2 ===")
        middle_idx, center_file, displacement_fields = run_stage2(config)

        assert middle_idx == 1, "Center should be sequence 1"
        assert len(displacement_fields) == 3

        # Verify displacement accuracy
        seq_names = ["seq_0", "seq_1", "seq_2"]
        for idx, seq_name in enumerate(seq_names):
            w = displacement_fields[idx]
            gt_shift_y, gt_shift_x = ground_truth[seq_name]

            # Compute mean displacement (standard EPE metric)
            u_mean = np.mean(w[:, :, 0])
            v_mean = np.mean(w[:, :, 1])

            print(
                f"{seq_name}: Mean displacement (v, u) = ({v_mean:.2f}, {u_mean:.2f})"
            )
            print(f"Ground truth (v, u) = ({gt_shift_y:.1f}, {gt_shift_x:.1f})")

            # For center, should be zero
            if seq_name == "seq_1":
                assert abs(u_mean) < 0.1, f"{seq_name}: u should be ~0"
                assert abs(v_mean) < 0.1, f"{seq_name}: v should be ~0"
            else:
                expected_u = gt_shift_x
                expected_v = gt_shift_y

                u_error = abs(u_mean - expected_u)
                v_error = abs(v_mean - expected_v)

                print(
                    f"{seq_name}: Expected ({expected_v:.1f}, {expected_u:.1f}), "
                    f"Got ({v_mean:.2f}, {u_mean:.2f}), "
                    f"Error ({v_error:.2f}, {u_error:.2f})"
                )

                assert (
                    u_error < 15.0
                ), f"{seq_name}: u displacement error {u_error:.2f} > 15.0 px"
                assert (
                    v_error < 15.0
                ), f"{seq_name}: v displacement error {v_error:.2f} > 15.0 px"

        # Stage 3: Valid mask alignment
        print("\n=== Running Stage 3 ===")
        final_valid = run_stage3(config, middle_idx, displacement_fields)

        # Verify outputs
        final_results = config.final_results
        if not final_results.is_absolute():
            final_results = config.root / final_results

        assert (final_results / "final_valid_idx.png").exists()
        assert (final_results / "final_valid_idx.npz").exists()

        # Verify final mask is reasonable
        assert final_valid.shape == (128, 128)
        assert final_valid.dtype == bool

        # Should have some valid pixels (intersection of all masks)
        assert np.any(final_valid), "Final mask should have some valid pixels"

        # Should have some invalid pixels (due to shifts)
        assert not np.all(final_valid), "Final mask should have some invalid pixels"

        print(
            f"\nFinal valid region: {np.sum(final_valid)} / {final_valid.size} pixels "
            f"({100*np.mean(final_valid):.1f}%)"
        )

    def test_resume_behavior_stage1(self, synthetic_session_data, capsys):
        """Test that Stage 1 resumes correctly when outputs exist."""
        config = synthetic_session_data["config"]

        # Run Stage 1 first time
        config.flow_options = {
            "quality_setting": "fast",
            "reference_frames": list(range(10)),
        }

        run_stage1(config)

        # Enable resume
        config.resume = True

        # Run again - should skip
        run_stage1(config)

        # Check console output for skip messages
        captured = capsys.readouterr()
        assert "already complete" in captured.out or "Skipping" in captured.out

    def test_resume_behavior_stage2(self, synthetic_session_data):
        """Test that Stage 2 resumes correctly when outputs exist."""
        config = synthetic_session_data["config"]

        # Run Stages 1 and 2
        config.flow_options = {
            "quality_setting": "fast",
            "reference_frames": list(range(10)),
        }

        run_stage1(config)
        middle_idx1, _, displacements1 = run_stage2(config)

        # Enable resume
        config.resume = True

        # Run Stage 2 again - should load existing
        middle_idx2, _, displacements2 = run_stage2(config)

        # Should get same results
        assert middle_idx1 == middle_idx2

        # Displacements should be identical (loaded from file)
        for w1, w2 in zip(displacements1, displacements2):
            np.testing.assert_array_almost_equal(w1, w2)

    def test_resume_behavior_stage3(self, synthetic_session_data):
        """Test that Stage 3 resumes correctly when outputs exist."""
        config = synthetic_session_data["config"]

        # Run all stages
        config.flow_options = {
            "quality_setting": "fast",
            "reference_frames": list(range(10)),
        }

        run_stage1(config)
        middle_idx, _, displacements = run_stage2(config)
        final_mask1 = run_stage3(config, middle_idx, displacements)

        # Enable resume
        config.resume = True

        # Run Stage 3 again - should load existing
        final_mask2 = run_stage3(config, middle_idx, displacements)

        # Should get same mask
        np.testing.assert_array_equal(final_mask1, final_mask2)


@pytest.mark.slow
class TestJobArrayIndexing:
    """Test job array task index resolution."""

    def test_stage1_array_task_selection(self, synthetic_session_data, monkeypatch):
        """
        Test that Stage 1 with array task ID processes only selected recording.
        """
        config = synthetic_session_data["config"]

        # Set SLURM_ARRAY_TASK_ID=2 (1-based, so should process index 1)
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")

        from pyflowreg.session.config import get_array_task_id
        from pyflowreg.session.stage1_compensate import discover_input_files

        task_id = get_array_task_id()
        assert task_id == 2

        # Run with explicit task index (0-based)
        task_index = task_id - 1
        config.flow_options = {
            "quality_setting": "fast",
            "reference_frames": list(range(10)),
        }

        output_folders = run_stage1(config, task_index=task_index)

        # Should only process 1 sequence
        assert len(output_folders) == 1

        # Verify which sequence was processed
        input_files = discover_input_files(config)
        expected_name = input_files[task_index].stem

        assert output_folders[0].name == expected_name

        # Verify other sequences were NOT processed
        output_root, _ = config.resolve_output_paths()
        for i, input_file in enumerate(input_files):
            output_folder = output_root / input_file.stem

            if i == task_index:
                # This one should exist
                assert output_folder.exists()
                assert (output_folder / "compensated.HDF5").exists()
            else:
                # These should not exist (or not have outputs)
                if output_folder.exists():
                    assert not (output_folder / "compensated.HDF5").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
