"""
Parallelization equivalence tests for z-align patch scoring.
"""

import numpy as np

from pyflowreg.z_align.pipeline import _estimate_z_patchwise


def test_estimate_z_patchwise_threading_matches_sequential():
    """Threaded patch scoring should match sequential output up to FP noise."""
    rng = np.random.default_rng(1234)
    H, W, C, Z, T = 32, 32, 1, 9, 7

    gx_vol = rng.standard_normal((H, W, C, Z), dtype=np.float32)
    gy_vol = rng.standard_normal((H, W, C, Z), dtype=np.float32)
    gx_f = rng.standard_normal((H, W, C, T), dtype=np.float32)
    gy_f = rng.standard_normal((H, W, C, T), dtype=np.float32)

    kwargs = {
        "anchor_z": 4,
        "win_half": 3,
        "patch_size": 8,
        "overlap": 0.5,
        "tau_scale": 1e-3,
        "z_smooth_sigma_spatial": 1.5,
        "z_smooth_sigma_temporal": 1.0,
    }

    z_seq = _estimate_z_patchwise(
        gx_vol,
        gy_vol,
        gx_f,
        gy_f,
        parallelization="sequential",
        n_jobs=1,
        **kwargs,
    )
    z_thr = _estimate_z_patchwise(
        gx_vol,
        gy_vol,
        gx_f,
        gy_f,
        parallelization="threading",
        n_jobs=4,
        **kwargs,
    )

    assert np.allclose(z_seq, z_thr, atol=1e-5, rtol=1e-5)
