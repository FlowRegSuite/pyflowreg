# Docs page : docs/api/session.md ("Stage 1: Per-Recording Compensation")
# Test      : tests/docs/api/test_session.py::TestSessionRunStage1Example
# Inputs    : session.toml and session/recording_*.tif -- created by the test harness
# [docs:start]
from pyflowreg.session import SessionConfig
from pyflowreg.session.stage1_compensate import run_stage1

config = SessionConfig.from_toml("session.toml")
config.n_workers = -1

# Override OFOptions directly on the config (or set via TOML/YAML)
config.flow_options = {
    "quality_setting": "fast",
    "save_w": True,
    "buffer_size": 1000,
}

output_folders = run_stage1(config)
# [docs:end]
