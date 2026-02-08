"""
Z-Shift Demo - MATLAB-style z alignment workflow

This example assumes the same input files as the MATLAB scripts:
- compensated.tiff         (time recording to z-correct)
- file_00004_00001.tif     (stack/source for reference volume creation)

Outputs (matching MATLAB names) are written to the working directory:
- aligned_stack/compensated.HDF5
- z_shift.HDF5
- compensated_shift_corrected.tif
- simulated_from_z.tif
"""

from pathlib import Path

from pyflowreg.z_align import ZAlignConfig, run_all_stages


def main():
    root = Path(".").resolve()

    required = [root / "compensated.tiff", root / "file_00004_00001.tif"]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input files in working directory: " + ", ".join(missing)
        )

    config = ZAlignConfig(
        root=root,
        # MATLAB-style inputs
        input_file="compensated.tiff",
        volume_input_file="file_00004_00001.tif",
        reference_source_file="compensated.tiff",
        # MATLAB script: read first 2000 frames with buffer/bin (10, 20)
        reference_source_frames=2000,
        reference_source_buffer_size=10,
        reference_source_bin_size=20,
        # Keep MATLAB output paths/names
        output_root=".",
        volume_output_dir="aligned_stack",
        z_shift_file="z_shift.HDF5",
        corrected_output_file="compensated_shift_corrected.tif",
        simulated_output_file="simulated_from_z.tif",
        # Stage toggles:
        # write_corrected=True  -> direct z-corrected signal
        # write_simulated=True  -> baseline simulated from z-shifts
        write_corrected=True,
        write_simulated=True,
        resume=True,
        # Stage 1 (volume build) defaults from MATLAB snippet
        stage1_alpha=5.0,
        stage1_quality_setting="quality",
        stage1_buffer_size=500,
        stage1_bin_size=1,
        stage1_update_reference=True,
        # Stage 2 (patch-based z estimation) defaults from MATLAB snippet
        input_buffer_size=50,
        input_bin_size=1,
        win_half=10,
        patch_size=128,
        overlap=0.75,
        spatial_sigma=1.5,
        temporal_sigma=1.5,
        z_smooth_sigma_spatial=5.0,
        z_smooth_sigma_temporal=1.5,
    )

    print("=" * 60)
    print("Z-SHIFT DEMO")
    print("=" * 60)
    print(f"Root: {root}")
    print("Input recording: compensated.tiff")
    print("Volume source: file_00004_00001.tif")

    outputs = run_all_stages(config)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Reference volume: {outputs['volume_path']}")
    print(f"Z-shift file:      {outputs['z_shift_path']}")
    if outputs["corrected_path"] is not None:
        print(f"Corrected signal:  {outputs['corrected_path']}")
    if outputs["simulated_path"] is not None:
        print(f"Simulated baseline:{outputs['simulated_path']}")


if __name__ == "__main__":
    main()
