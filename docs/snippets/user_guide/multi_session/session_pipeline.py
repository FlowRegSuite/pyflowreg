# Docs page : docs/user_guide/multi_session.md ("Run Processing")
# Test      : tests/docs/user_guide/test_multi_session.py::TestMultiSessionPipeline
# Inputs    : session/recording_000.tif, recording_001.tif, recording_002.tif -- created by the test harness
# [docs:start]
from pyflowreg.session import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1
from pyflowreg.session.stage2_between_avgs import run_stage2
from pyflowreg.session.stage3_valid_mask import run_stage3

# Point the session at the directory holding your recordings. The same
# settings can be loaded from a file with SessionConfig.from_toml("session.toml").
config = SessionConfig(
    root="session",
    pattern="recording_*.tif",
    stage1_quality_setting="fast",
)

# Run all three stages
output_folders = run_stage1(config)
middle_idx, center_file, displacements = run_stage2(config)
final_mask = run_stage3(config, middle_idx, displacements)
# [docs:end]
