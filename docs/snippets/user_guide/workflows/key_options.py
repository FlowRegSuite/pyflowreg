# Docs page : docs/user_guide/workflows.md ("Key Configuration Options")
# Test      : tests/docs/user_guide/test_workflows.py::TestWorkflowsKeyOptions
# Inputs    : none
#
# Page context (not rendered): OFOptions and OutputFormat are imported from
# pyflowreg.motion_correction as in the examples earlier on the page.
from pyflowreg.motion_correction import OFOptions, OutputFormat

# [docs:start]
options = OFOptions(
    output_format=OutputFormat.NULL,  # See File Formats page for all values
    buffer_size=20,  # Frames per batch
    save_w=True,  # Save displacement fields
    levels=5,  # Maximum number of pyramid levels
    iterations=50,  # Iterations per level
    quality_setting="balanced",  # "fast", "balanced", "quality"
)
# [docs:end]
