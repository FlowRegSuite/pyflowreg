# Docs page : docs/user_guide/3d_volumes.md ("Post-Processing: Frame Binning")
# Test      : tests/docs/user_guide/test_3d_volumes.py::TestThreeDVolumesManualBinning
# Inputs    : aligned_sequence/compensated.HDF5 -- created by the test harness
# [docs:start]
from pyflowreg.util.io.factory import get_video_file_reader

# Read all registered frames through the PyFlowReg I/O system
reader = get_video_file_reader("aligned_sequence/compensated.HDF5")
registered = reader[:]  # (T, H, W, C) where T = Z * frames_per_slice
reader.close()

frames_per_slice = 9
T, H, W, C = registered.shape
n_slices = T // frames_per_slice

# Reshape and average the repetitions of each z-slice
volume = registered.reshape(n_slices, frames_per_slice, H, W, C)
volume = volume.mean(axis=1)  # (Z, H, W, C)
# [docs:end]
