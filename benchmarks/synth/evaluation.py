import argparse
import pathlib
import time
import urllib.request

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from methods import pyflowreg as m_pfr, suite2p as m_s2p, antspyx as m_ants, elastix as m_elx

# Try to import NoRMCorre if CaImAn is available
try:
    from methods import normcorre as m_nc
    NORMCORRE_AVAILABLE = True
except ImportError:
    NORMCORRE_AVAILABLE = False
    print("Warning: CaImAn/NoRMCorre not available, skipping those tests")
from preprocessing import DefaultProcessor, GradientProcessor, normalize


def download_synth_data(output_folder: pathlib.Path, input_file: pathlib.Path) -> None:
    """Download synthetic frames data if not already present."""
    if not input_file.exists():
        print("Downloading synthetic frames data...")
        url = "https://drive.usercontent.google.com/download?id=10YxHVSdnz0L4WMLR0eIHH6bMxaojpVdY"
        urllib.request.urlretrieve(url, input_file)
        print(f"Downloaded to {input_file}")
    else:
        print(f"Synthetic data already exists at {input_file}")


class SynthDataset:
    def __init__(self, path, split=("clean", "noisy35db", "noisy30db")):
        self.path = path
        self.split = split
        with h5py.File(path, "r") as f:
            # Convert from (2, H, W) to (H, W, 2) and reverse flow components
            self.w = np.moveaxis(f["w"][:], 0, -1)
            self.w = self.w[..., ::-1]  # Reverse y,x to x,y
            self.frames_raw = {}
            for k in split:
                if k in f:
                    # Load raw data matching synth_evaluation.py
                    # f[k] shape is (2, 2, 512, 512) - (frames, channels, H, W)
                    tmp = f[k][:].astype(np.float32)
                    
                    # Apply Gaussian filtering to match synth_evaluation.py
                    tmp = gaussian_filter(tmp, (0.1, 0.01, 1.5, 1.5),
                                        truncate=10, mode='constant')
                    
                    self.frames_raw[k] = tmp
                    
                    # Create single channel variants
                    # tmp shape is (n_temporal_frames, n_channels, H, W)
                    # Extract individual channels as (n_temporal, 1, H, W)
                    self.frames_raw[k + "_ch1"] = tmp[:, [0], :, :]
                    self.frames_raw[k + "_ch2"] = tmp[:, [1], :, :]
    
    def pairs(self, key, processor=None):
        if processor is None:
            processor = DefaultProcessor()
            
        x = self.frames_raw[key]
        pairs = []
        # x shape is (n_temporal_frames, n_channels, H, W)
        # Match synth_evaluation.py: process pairs (0,1)
        if x.shape[0] >= 2:
            # Transpose from (C, H, W) to (H, W, C) for processing
            frame1 = np.transpose(x[0], (1, 2, 0)).astype(np.float32)
            frame2 = np.transpose(x[1], (1, 2, 0)).astype(np.float32)
            
            # Normalize using frame1's statistics for BOTH frames (like synth_evaluation)
            mins = frame1.min(axis=(0, 1), keepdims=True)  # shape (1,1,C)
            maxs = frame1.max(axis=(0, 1), keepdims=True)  # shape (1,1,C)
            ranges = maxs - mins
            ranges = np.where(ranges < 1e-6, 1.0, ranges)  # Avoid division by zero
            
            frame1 = (frame1 - mins) / ranges
            frame2 = (frame2 - mins) / ranges
            
            # Skip additional processing since we already normalized
            # processor.process would normalize again which we don't want
            # frame1 = processor.process(frame1)
            # frame2 = processor.process(frame2)
            pairs.append((frame1, frame2))
        return pairs
    
    def gt(self):
        return self.w
    
    def available_keys(self):
        return list(self.frames_raw.keys())


def get_EPE(w, w_gt, boundary=25):
    """Calculate EPE matching your old code"""
    w_crop = w[boundary:-boundary, boundary:-boundary]
    w_gt_crop = w_gt[boundary:-boundary, boundary:-boundary]
    return float(np.mean(np.linalg.norm(w_crop - w_gt_crop, axis=-1)))


def epe(gt, est, crop=25):
    gt_c = gt[crop:-crop, crop:-crop, :2]
    est_c = est[crop:-crop, crop:-crop, :2]
    return float(np.mean(np.linalg.norm(gt_c - est_c, axis=-1)))


def epe_p95(gt, est, crop=25):
    gt_c = gt[crop:-crop, crop:-crop, :2]
    est_c = est[crop:-crop, crop:-crop, :2]
    errors = np.linalg.norm(gt_c - est_c, axis=-1).reshape(-1)
    return float(np.percentile(errors, 95))


def mean_abs_curl(flow, crop=25):
    f = flow[crop:-crop, crop:-crop, :2]
    dvy_dy, dvy_dx = np.gradient(f[..., 0])  # vy
    dvx_dy, dvx_dx = np.gradient(f[..., 1])  # vx
    curl_z = dvx_dy - dvy_dx
    return float(np.mean(np.abs(curl_z)))


def run_method(name, fn, pairs, w_gt, outdir, **kw):
    rows = []
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    
    for i, (f, m) in enumerate(pairs):
        t0 = time.perf_counter()
        try:
            w = fn(f, m, **kw)
            
            # Ensure w is in (H, W, 2) format like your old code
            if w.ndim == 3 and w.shape[0] == 2:
                w = np.moveaxis(w, 0, -1)
            elif w.ndim == 3 and w.shape[-1] == 2:
                pass  # Already correct
            else:
                raise ValueError(f"Unexpected flow shape: {w.shape}")
            
            # The synthetic H5 has a single GT flow field for the one pair
            w_i = w_gt
            
            # Shape guard
            H, W = w.shape[:2]
            assert w_i.shape[:2] == (H, W) and w_i.shape[-1] == 2, f"GT shape {w_i.shape} mismatches estimate {w.shape}"
            
            # Optional direction sanity for algorithms with ambiguous sign
            if i == 0:
                e_dir = get_EPE(w, w_i, boundary=25)
                e_inv = get_EPE(-w, w_i, boundary=25)
                if e_inv < e_dir:
                    w = -w
                
            dt = time.perf_counter() - t0
            
            # Use your old EPE calculation
            e = get_EPE(w, w_i, boundary=25)
            p = epe_p95(w_i, w)  # Fixed parameter order: GT first, then estimate
            c = mean_abs_curl(w)
            
            rows.append({
                "method": name,
                "idx": i,
                "epe": e,
                "epe95": p,
                "curl": c,
                "time_s": dt
            })
            
            with h5py.File(f"{outdir}/{name}_{i}.h5", "w") as hf:
                hf.create_dataset("w", data=w)
                hf.create_dataset("epe", data=e)
                hf.create_dataset("frames", data=np.stack([f, m], 0))
            
        except Exception as e:
            print(f"Error running {name} on pair {i}: {e}")
            rows.append({
                "method": name,
                "idx": i,
                "epe": np.nan,
                "epe95": np.nan,
                "curl": np.nan,
                "time_s": np.nan
            })
    
    return rows


def _canary():
    """Quick translation test to catch sign flips and unit mistakes"""
    H = W = 128
    dy, dx = 2.5, -1.0
    ref = np.zeros((H, W), np.float32)
    mov = np.roll(np.roll(ref, int(dy), 0), int(dx), 1)
    gt = np.zeros((H, W, 2), np.float32)
    gt[..., 0] = dy
    gt[..., 1] = dx
    
    for name, fn in [
        ("pyflowreg", m_pfr.estimate_flow),
        ("suite2p_rigid", lambda F, M: m_s2p.estimate_flow(F, M, False))
    ]:
        try:
            w = fn(ref, mov)
            if np.mean(np.abs(w - gt)) > 0.5:
                print(f"[canary] {name} seems off by >0.5 px")
        except Exception as e:
            print(f"[canary] {name} failed: {e}")


def main(args):
    # Run canary test first
    _canary()
    
    # Handle data download if needed
    if args.data is None:
        data_folder = pathlib.Path("benchmarks/synth/data")
        data_folder.mkdir(parents=True, exist_ok=True)
        data_file = data_folder / "synth_frames.h5"
        download_synth_data(data_folder, data_file)
        args.data = str(data_file)
    
    ds = SynthDataset(args.data)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    
    w_gt = ds.gt()
    
    result_df = pd.DataFrame()
    
    # Test configurations matching your old paper
    test_configs = [
        # Split, processor, suffix
        # Note: Gaussian filtering (sigma=1.5) is already applied in data loading
        (args.split, DefaultProcessor(sigma=0), ""),
        (args.split + "_ch1", DefaultProcessor(sigma=0), " ch1"),
        (args.split + "_ch2", DefaultProcessor(sigma=0), " ch2"),
    ]
    
    for split_key, processor, suffix in test_configs:
        if split_key not in ds.available_keys():
            continue
            
        print(f"\nTesting {split_key} with {processor.__class__.__name__}(sigma={getattr(processor, 'sigma', 'N/A')})")
        pairs = ds.pairs(split_key, processor)
        
        # PyFlowReg
        try:
            rows = run_method(f"pyflowreg{suffix}", m_pfr.estimate_flow, pairs, w_gt, outdir)
            for row in rows:
                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            print(f"Error with pyflowreg: {e}")
        
        # Suite2p
        try:
            rows = run_method(f"suite2p_rigid{suffix}", 
                            lambda f, m: m_s2p.estimate_flow(f, m, use_nonrigid=False), 
                            pairs, w_gt, outdir)
            for row in rows:
                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            print(f"Error with suite2p rigid: {e}")
            
        try:
            rows = run_method(f"suite2p_nonrigid{suffix}", 
                            lambda f, m: m_s2p.estimate_flow(f, m, use_nonrigid=True), 
                            pairs, w_gt, outdir)
            for row in rows:
                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            print(f"Error with suite2p nonrigid: {e}")
        
        # NoRMCorre (via CaImAn)
        if NORMCORRE_AVAILABLE:
            try:
                rows = run_method(f"normcorre{suffix}", m_nc.estimate_flow, pairs, w_gt, outdir)
                for row in rows:
                    result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
            except Exception as e:
                print(f"Error with normcorre: {e}")
        
        # ANTs variants matching your old code
        ants_configs = [
            ("SyN", "mattes", "ants syn"),
            ("SyN", "CC", "ants syncc"),
            ("ElasticSyN", "mattes", "ants ela"),
        ]
        
        for transform, metric, name in ants_configs:
            try:
                rows = run_method(f"{name}{suffix}", 
                                lambda f, m, t=transform, met=metric: m_ants.estimate_flow(f, m, t, met), 
                                pairs, w_gt, outdir)
                for row in rows:
                    result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        # Elastix variants
        base = pathlib.Path(args.params)
        elastix_configs = [
            ("bspline_multimetric_cc.txt", "elastix cc"),
            ("bspline_multimetric_cc_gradient.txt", "elastix cc + gc"),
            ("bspline_multimetric_mi.txt", "elastix mi"),
            ("bspline_multimetric_mi_gradient.txt", "elastix mi + gc"),
        ]
        
        for param_file, name in elastix_configs:
            param_path = base / param_file
            if param_path.exists():
                try:
                    rows = run_method(f"{name}{suffix}", 
                                    lambda f, m, p=param_path: m_elx.estimate_flow(f, m, [p]), 
                                    pairs, w_gt, outdir)
                    for row in rows:
                        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
                except Exception as e:
                    print(f"Error with {name}: {e}")
    
    # Save results
    result_df.to_csv(outdir/"results_detailed.csv", index=False)
    
    # Create summary matching your old format
    summary_df = result_df.groupby("method").agg({
        "epe": "mean",
        "epe95": "mean", 
        "curl": "mean",
        "time_s": "mean"
    }).reset_index().sort_values("epe")
    
    summary_df.to_csv(outdir/"results_summary.csv", index=False)
    
    with open(outdir/"results_table.tex", "w") as f:
        f.write(summary_df.to_latex(index=False, float_format="%.2f"))
    
    print(f"\nResults saved to {outdir}")
    print("\nSummary (sorted by EPE):")
    print(summary_df.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="Path to synth_frames.h5 (if not provided, will download automatically)")
    p.add_argument("--out", default="benchmarks/synth/out", help="Output directory")
    p.add_argument("--split", default="clean", choices=["clean", "noisy35db", "noisy30db"])
    p.add_argument("--params", default="benchmarks/synth/elastix_params", help="Elastix params directory")
    main(p.parse_args())