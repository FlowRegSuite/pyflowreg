# Docs page : docs/user_guide/online_processing.md ("Complete Example")
# Test      : tests/docs/user_guide/test_online_processing.py::TestOnlineProcessingStreamingLoop
# Inputs    : recording.tif -- created by the test harness
# [docs:start]
import numpy as np

from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.motion_correction.flow_reg_live import FlowRegLive
from pyflowreg.util.io import get_video_file_reader

# Read a short recording into memory as (T, H, W, C)
reader = get_video_file_reader("recording.tif")
video = reader[:]
reader.close()

# Frames are normalized against the raw reference's intensity range
# internally, so they can stay at the recording's native scale (uint16
# here); cast to float32 only to keep the demo arithmetic simple
video = video.astype(np.float32)

# Configure optical flow; quality_setting is forced to "fast" internally
options = OFOptions(
    alpha=4,  # stronger regularization
    sigma=[[2.0, 2.0, 0.5], [2.0, 2.0, 0.5]],  # spatial + temporal filtering
    levels=100,
    iterations=50,
    eta=0.8,
    channel_normalization="separate",
)

flow_reg = FlowRegLive(
    options=options,
    reference_buffer_size=50,
    reference_update_interval=20,
    reference_update_weight=0.2,
)

# Initialize the reference from the first frames of the recording
flow_reg.set_reference(video[:10])

# Stream frames through the corrector one at a time
for frame in video:
    registered, flow = flow_reg(frame)
    # registered: (H, W, C) corrected frame
    # flow: (H, W, 2) displacement field (u = flow[..., 0], v = flow[..., 1])
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    print(f"max displacement: {magnitude.max():.2f} px")
# [docs:end]
