# Docs page : docs/quickstart.md ("Multi-Session Processing")
# Test      : tests/docs/test_quickstart.py::TestQuickstartMultiSession
# Inputs    : session/recording_*.tif -- created by the test harness
# [docs:start]
from pyflowreg.session import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1
from pyflowreg.session.stage2_between_avgs import run_stage2
from pyflowreg.session.stage3_valid_mask import run_stage3

# Configure session
config = SessionConfig(
    root="session/",  # Directory containing the recordings
    pattern="recording_*.tif",
    output_root="compensated",
    n_workers=-1,  # Stage 1 workers (-1 = all CPU cores)
    resume=True,  # Enable resume for crash safety
)

# Stage 1: Motion correct each recording
output_folders = run_stage1(config)

# Stage 2: Align recordings to common reference
middle_idx, center_file, displacements = run_stage2(config)

# Stage 3: Compute valid mask across all recordings
final_mask = run_stage3(config, middle_idx, displacements)
# [docs:end]
