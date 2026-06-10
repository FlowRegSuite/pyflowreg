# Docs page : docs/quickstart.md ("Configuration Options" / "Key Parameters")
# Test      : tests/docs/test_quickstart.py::TestQuickstartKeyParameters
# Inputs    : none
# [docs:start]
from pyflowreg.motion_correction import OFOptions

options = OFOptions(
    # Flow parameters
    alpha=4,  # Smoothness regularization weight; default (1.5, 1.5),
    # scalars are expanded to a 2-tuple
    iterations=50,  # SOR iterations per pyramid level (default 50)
    levels=100,  # Maximum number of pyramid levels (default 100)
    eta=0.8,  # Pyramid downsampling factor (default 0.8)
    min_level=-1,  # Finest pyramid level to solve; -1 (default) derives it
    # from quality_setting, values >= 0 override the preset
    # and switch quality_setting to "custom"
    # Preprocessing
    bin_size=1,  # Temporal binning factor (default 1)
    sigma=[1.0, 1.0, 0.1],  # Gaussian filter sigma [sx, sy, st]; a single triple
    # is applied to all channels (the default)
    # Reference
    reference_frames=[0, 1, 2, 3, 4],  # Frames to average for reference
    # (default list(range(50, 500)))
    # I/O
    buffer_size=400,  # Frames read per batch (default 400)
    output_format="HDF5",  # Output format (default "MAT")
    save_w=True,  # Save displacement fields (default False)
)
# [docs:end]
