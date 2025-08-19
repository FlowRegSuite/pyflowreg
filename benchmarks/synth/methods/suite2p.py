import numpy as np
import tempfile
import pathlib
import tifffile as tiff
import cv2
from suite2p.run_s2p import run_s2p


def _ops(h, w, nrigid):
    return dict(
        data_path=[],
        save_path0="",
        nplanes=1,
        nchannels=1,
        do_registration=1,
        nonrigid=nrigid,
        two_step_registration=True,
        block_size=[128, 128],
        smooth_sigma=1.15,
        maxregshift=0.1,
        maxregshiftNR=10,
        keep_movie_raw=True,
        roidetect=False
    )


def estimate_flow(fixed, moving, nonrigid=False):
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d)
        
        if fixed.ndim == 3:
            fixed = fixed[..., 0]
        if moving.ndim == 3:
            moving = moving[..., 0]
        
        tiff.imwrite(p/"stack.tif", np.stack([fixed, moving], 0).astype(np.float32))
        
        ops = _ops(*fixed.shape, int(nonrigid))
        ops["save_path0"] = str(p)
        db = {
            "h5py": [],
            "look_one_level_down": False,
            "data_path": [str(p/"stack.tif")]
        }
        
        run_s2p(ops, db)
        
        # Find ops.npy with plane0 fallback
        ops_path = p/"ops.npy"
        if not ops_path.exists():
            # Check common plane subdirectory locations
            for plane_dir in [p/"plane0", p/"plane00", p/"suite2p"/"plane0"]:
                candidate = plane_dir/"ops.npy"
                if candidate.exists():
                    ops_path = candidate
                    break
        
        ops_out = np.load(ops_path, allow_pickle=True).item()
        H, W = fixed.shape
        v = np.zeros((2, H, W), np.float32)
        
        if nonrigid:
            # Extract nonrigid shifts for second frame
            y1 = np.array(ops_out["yoff1"])[1]
            x1 = np.array(ops_out["xoff1"])[1]
            # Expand the grid to full resolution
            v[0] = cv2.resize(y1, (W, H), interpolation=cv2.INTER_NEAREST)
            v[1] = cv2.resize(x1, (W, H), interpolation=cv2.INTER_NEAREST)
            return v
        
        # Rigid registration
        y = np.array(ops_out["yoff"])[1]
        x = np.array(ops_out["xoff"])[1]
        v[0] += y
        v[1] += x
        return v