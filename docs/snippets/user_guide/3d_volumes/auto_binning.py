# Docs page : docs/user_guide/3d_volumes.md ("Post-Processing: Frame Binning")
# Test      : tests/docs/user_guide/test_3d_volumes.py::TestThreeDVolumesAutoBinning
# Inputs    : aligned_sequence/compensated.HDF5 -- created by the test harness
# [docs:start]
import numpy as np
from pyflowreg.util.io.factory import get_video_file_reader

# The reader can perform binning automatically
reader = get_video_file_reader(
    "aligned_sequence/compensated.HDF5",
    buffer_size=100,
    bin_size=9,  # Bin every 9 frames
)

# This returns already-binned data where each "frame" is the average of 9 registered frames
binned_volume = []
while reader.has_batch():
    batch = reader.read_batch()
    binned_volume.append(batch)
reader.close()

binned_volume = np.concatenate(binned_volume, axis=0)
# [docs:end]
