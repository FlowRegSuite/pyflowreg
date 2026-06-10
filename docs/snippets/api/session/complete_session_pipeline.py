# Docs page : docs/api/session.md ("Complete Example")
# Test      : tests/docs/api/test_session.py::TestSessionCompleteExample
# Inputs    : session/recording_*.tif -- created by the test harness
# [docs:start]
import numpy as np

from pyflowreg.session import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1
from pyflowreg.session.stage2_between_avgs import run_stage2
from pyflowreg.session.stage3_valid_mask import run_stage3

# Configure session
config = SessionConfig(
    root="session/",
    pattern="recording_*.tif",
    center="recording_001.tif",
    output_root="compensated",
    final_results="results",
    resume=True,
    n_workers=-1,
    flow_backend="flowreg",
    cc_upsample=4,
    sigma_smooth=6.0,
    alpha_between=25.0,
    iterations_between=100,
    stage2_constancy_assumption="gc",
)

# Stage 1: Motion correct each recording
print("Running Stage 1...")
config.flow_options = {
    "quality_setting": "balanced",
    "constancy_assumption": "gc",
    "save_valid_idx": True,
    "save_w": False,
}
output_folders = run_stage1(config)

# Stage 2: Align temporal averages
print("Running Stage 2...")
middle_idx, center_file, displacement_fields = run_stage2(config)

# Stage 3: Compute final valid mask
print("Running Stage 3...")
final_mask = run_stage3(config, middle_idx, displacement_fields)

print(f"Final valid region: {np.sum(final_mask)} pixels")
# [docs:end]
