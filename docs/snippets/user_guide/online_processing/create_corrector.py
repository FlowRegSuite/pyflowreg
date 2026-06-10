# Docs page : docs/user_guide/online_processing.md ("Creating a Corrector")
# Test      : tests/docs/user_guide/test_online_processing.py::TestOnlineProcessingCreateCorrector
# Inputs    : none
# [docs:start]
from pyflowreg.motion_correction.flow_reg_live import FlowRegLive

flow_reg = FlowRegLive(
    options=None,  # OFOptions, or None for fast defaults
    reference_buffer_size=50,  # frames collected before set_reference()
    reference_update_interval=20,  # mix a new frame into the reference every N frames
    reference_update_weight=0.2,  # blend weight for reference updates
    truncate=4.0,  # Gaussian kernels truncated at this many sigmas
)
# [docs:end]
