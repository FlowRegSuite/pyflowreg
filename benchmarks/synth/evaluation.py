import argparse
import pathlib
import time
import urllib.request

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from methods import pyflowreg as m_pfr, suite2p as m_s2p, jnormcorre as m_jnc, antspyx as m_ants, elastix as m_elx
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
            self.w = np.swapaxes(f["w"][:], 1, 2)  # Match your old format
            self.frames_raw = {}
            for k in split:
                if k in f:
                    # Match your old preprocessing: swapaxes and normalize
                    tmp = np.swapaxes(f[k][:], 2, 3).astype(np.float32) / 4095
                    # Channel-wise normalization like your old code
                    for i in range(tmp.shape[1]):
                        tmp[:, i] = normalize(tmp[:, i])
                    self.frames_raw[k] = tmp
                    
                    # Create single channel variants
                    self.frames_raw[k + "_ch1"] = tmp[:, [0]]
                    self.frames_raw[k + "_ch2"] = tmp[:, [1]]
    
    def pairs(self, key, processor=None):
        if processor is None:
            processor = DefaultProcessor()
            
        x = self.frames_raw[key]
        pairs = []
        for i in range(x.shape[0]):
            frame1 = processor.process(x[i, 0])
            frame2 = processor.process(x[i, 1])
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
    flow_c = flow[crop:-crop, crop:-crop, :2]
    fy, fx = np.gradient(flow_c[..., 0])  # vy gradients
    uy, ux = np.gradient(flow_c[..., 1])  # vx gradients
    curl = np.abs(uy - fx)  # Correct 2D curl: ∂vx/∂y - ∂vy/∂x
    return float(np.mean(curl))


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
            
            # Index the correct GT for this pair
            w_i = w_gt[i]
                
            dt = time.perf_counter() - t0
            
            # Use your old EPE calculation
            e = get_EPE(w, w_i, boundary=25)
            p = epe_p95(w, w_i)
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


def main(args):
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
        (args.split, DefaultProcessor(sigma=0), ""),
        (args.split, DefaultProcessor(sigma=1.5), "*"),
        (args.split, GradientProcessor(sigma=1.5), " gc*"),
        (args.split + "_ch1", DefaultProcessor(sigma=0), " ch1"),
        (args.split + "_ch1", DefaultProcessor(sigma=1.5), " ch1*"),
        (args.split + "_ch2", DefaultProcessor(sigma=0), " ch2"),
        (args.split + "_ch2", DefaultProcessor(sigma=1.5), " ch2*"),
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
                            lambda f, m: m_s2p.estimate_flow(f, m, False), 
                            pairs, w_gt, outdir)
            for row in rows:
                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            print(f"Error with suite2p rigid: {e}")
            
        try:
            rows = run_method(f"suite2p_nonrigid{suffix}", 
                            lambda f, m: m_s2p.estimate_flow(f, m, True), 
                            pairs, w_gt, outdir)
            for row in rows:
                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            print(f"Error with suite2p nonrigid: {e}")
        
        # jNormCorre
        try:
            rows = run_method(f"jnormcorre{suffix}", m_jnc.estimate_flow, pairs, w_gt, outdir)
            for row in rows:
                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            print(f"Error with jnormcorre: {e}")
        
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