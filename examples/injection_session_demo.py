"""
Injection Session Demo - Multi-recording session processing demonstration

Creates synthetic multi-file dataset by shifting the injection demo video,
then runs the complete 3-stage session processing pipeline.

Stage 1: Per-recording motion correction
Stage 2: Inter-sequence alignment
Stage 3: Valid mask computation
"""

from pathlib import Path
import numpy as np

from pyflowreg.util.download import download_demo_data
from pyflowreg.util.io.factory import get_video_file_reader, get_video_file_writer
from pyflowreg.session.config import SessionConfig
from pyflowreg.session.cli import run_all_stages


def create_shifted_video(video, shift_x, shift_y, crop_border=50):
    """
    Create shifted and cropped version of video.

    Parameters
    ----------
    video : ndarray, shape (T, H, W, C)
        Input video
    shift_x : int
        Horizontal shift (positive = right)
    shift_y : int
        Vertical shift (positive = down)
    crop_border : int
        Pixels to crop from all sides

    Returns
    -------
    shifted_video : ndarray
        Shifted and cropped video
    """
    T, H, W, C = video.shape

    # Create shifted version
    shifted = np.zeros_like(video)

    # Calculate source and destination regions
    # Positive shift_x means content moves right, so we copy from left
    src_x_start = max(0, -shift_x)
    src_x_end = min(W, W - shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end = min(W, W + shift_x)

    # Positive shift_y means content moves down, so we copy from top
    src_y_start = max(0, -shift_y)
    src_y_end = min(H, H - shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end = min(H, H + shift_y)

    # Apply shift
    shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = video[
        :, src_y_start:src_y_end, src_x_start:src_x_end, :
    ]

    # Crop borders
    cropped = shifted[:, crop_border:-crop_border, crop_border:-crop_border, :]

    return cropped


def main():
    # Setup paths
    demo_folder = Path("injection_session_demo")
    demo_folder.mkdir(exist_ok=True)

    print("=" * 60)
    print("INJECTION SESSION DEMO")
    print("=" * 60)

    # Step 1: Download injection demo data
    print("\nStep 1: Downloading injection demo data...")
    input_file = download_demo_data("injection.tiff")
    print(f"Downloaded to: {input_file}")

    # Step 2: Read video
    print("\nStep 2: Reading video...")
    reader = get_video_file_reader(str(input_file))
    video = reader[:]
    reader.close()

    print(f"Video shape: {video.shape}")
    print(f"Video dtype: {video.dtype}")

    # Step 3: Create shifted variants
    print("\nStep 3: Creating shifted variants...")

    # injection_0: shifted by -50, -50 (left, up)
    print("  Creating injection_0.tif (shift -50, -50)...")
    video_0 = create_shifted_video(video, shift_x=-50, shift_y=-50, crop_border=50)

    # injection_1: original, cropped
    print("  Creating injection_1.tif (original, cropped)...")
    video_1 = video[:, 50:-50, 50:-50, :]

    # injection_2: shifted by +50, +50 (right, down)
    print("  Creating injection_2.tif (shift +50, +50)...")
    video_2 = create_shifted_video(video, shift_x=50, shift_y=50, crop_border=50)

    print(f"  Shifted video shapes: {video_0.shape}, {video_1.shape}, {video_2.shape}")

    # Step 4: Write variants to demo folder
    print("\nStep 4: Writing variants to disk...")

    for idx, video_data in enumerate([video_0, video_1, video_2]):
        output_path = demo_folder / f"injection_{idx}.tif"
        print(f"  Writing {output_path.name}...")

        writer = get_video_file_writer(str(output_path), output_format="TIFF")
        writer.write_frames(video_data)
        writer.close()

    # Step 5: Delete original injection.tif if it exists in demo folder
    original_in_demo = demo_folder / "injection.tif"
    if original_in_demo.exists():
        print(f"\nStep 5: Deleting {original_in_demo}...")
        original_in_demo.unlink()

    print("\nData preparation complete!")
    print(f"Created 3 files in {demo_folder}:")
    for f in sorted(demo_folder.glob("injection_*.tif")):
        print(f"  - {f.name}")

    # Step 6: Create session configuration
    print("\n" + "=" * 60)
    print("RUNNING SESSION PROCESSING")
    print("=" * 60)

    config = SessionConfig(
        root=demo_folder,
        pattern="injection_*.tif",
        output_root="compensated_outputs",
        final_results="final_results",
        resume=True,
        scheduler="local",
        flow_backend="flowreg",
        flow_options={
            "quality_setting": "balanced",
            "bin_size": 5,
            "buffer_size": 1000,
        },
        cc_upsample=1,
        sigma_smooth=6.0,
        alpha_between=25.0,
        iterations_between=100,
    )

    print("\nSession configuration:")
    print(f"  Root: {config.root}")
    print(f"  Pattern: {config.pattern}")
    print(f"  Output root: {config.output_root}")
    print(f"  Final results: {config.final_results}")

    # Step 7: Run all stages
    print("\n" + "=" * 60)
    print("Running session pipeline (all 3 stages)...")
    print("=" * 60)

    run_all_stages(config)

    print("\n" + "=" * 60)
    print("SESSION DEMO COMPLETE")
    print("=" * 60)
    print("\nResults saved to:")
    print(f"  Per-recording outputs: {demo_folder / config.output_root}")
    print(f"  Final session results: {demo_folder / config.final_results}")
    print("\nCheck final_valid_idx.png for the session-wide valid mask!")


if __name__ == "__main__":
    main()
