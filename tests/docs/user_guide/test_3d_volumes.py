"""
Executable-documentation tests for docs/user_guide/3d_volumes.md.

The "Post-Processing: Frame Binning" section shows two ways to collapse the
repeated frames acquired per z-slice into a volume: the reader's automatic
temporal binning driven by the sequential ``has_batch()`` / ``read_batch()``
loop, and a manual ``reader[:]`` + reshape. Both extracted snippets read a
``aligned_sequence/compensated.HDF5`` file (the output of the registration
step earlier on the page); the harness materializes that file with a tiny
synthetic stack of 27 frames (3 z-slices x 9 repetitions).

The full ``compensate_recording`` registration blocks on the same page are
left inline (allowlisted) because they run the motion-correction pipeline and
are not fast enough for the docs test budget.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.docs_example

# 3 z-slices x 9 repetitions per slice -> divisible by frames_per_slice=9.
FRAMES_PER_SLICE = 9
N_SLICES = 3
STACK_SHAPE = (N_SLICES * FRAMES_PER_SLICE, 32, 48, 1)


class TestThreeDVolumesAutoBinning:
    """docs/user_guide/3d_volumes.md, "Post-Processing: Frame Binning" (reader binning)."""

    def test_auto_binning_executes(self, materialize_input, snippet_runner):
        # The snippet reads this exact path with bin_size=9.
        materialize_input("aligned_sequence/compensated.HDF5", shape=STACK_SHAPE)

        ns = snippet_runner("user_guide/3d_volumes/auto_binning.py")

        # bin_size=9 over 27 raw frames -> 3 binned z-slices (Z, H, W, C).
        binned_volume = ns["binned_volume"]
        assert isinstance(binned_volume, np.ndarray)
        assert binned_volume.shape == (N_SLICES, 32, 48, 1)
        # Each binned frame is the average of 9 registered frames; binning
        # preserves the source dtype (see VideoReader.bin_frames).
        assert binned_volume.dtype == np.uint16


class TestThreeDVolumesManualBinning:
    """docs/user_guide/3d_volumes.md, "Post-Processing: Frame Binning" (manual reshape)."""

    def test_manual_binning_executes(self, materialize_input, snippet_runner):
        materialize_input("aligned_sequence/compensated.HDF5", shape=STACK_SHAPE)

        ns = snippet_runner("user_guide/3d_volumes/manual_binning.py")

        # reader[:] returns all 27 frames; reshape(3, 9, ...).mean(axis=1)
        # collapses each slice's repetitions into one (Z, H, W, C) volume.
        assert ns["frames_per_slice"] == FRAMES_PER_SLICE
        assert ns["n_slices"] == N_SLICES
        assert ns["registered"].shape == STACK_SHAPE

        volume = ns["volume"]
        assert isinstance(volume, np.ndarray)
        assert volume.shape == (N_SLICES, 32, 48, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
