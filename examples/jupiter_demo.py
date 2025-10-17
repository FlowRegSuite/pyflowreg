"""
Jupiter Demo - Motion Compensation Example
Author: Philipp Flotho (Python port)
Copyright 2021 by Philipp Flotho, All rights reserved.

This example downloads jupiter demo data and demonstrates minimal motion compensation config.
"""

from pathlib import Path
import numpy as np
import cv2
from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.motion_correction.compensate_recording import compensate_recording
from pyflowreg.util.io.factory import get_video_file_reader
from pyflowreg.util.download import download_demo_data


def main():
    # Prepare output directory for results
    output_folder = Path("jupiter_demo")
    output_folder.mkdir(exist_ok=True)

    # Download data to data/ folder (default location)
    input_file = download_demo_data("jupiter.tiff")

    # Create OF_options matching MATLAB configuration
    options = OFOptions(
        input_file=str(input_file),
        output_path=str(output_folder / "hdf5_comp_minimal"),
        output_format="HDF5",
        alpha=4,  # Larger alpha to avoid registering changing morphology
        quality_setting="balanced",  # Default in MATLAB is 'quality'
        output_typename="",
        reference_frames=list(
            range(100, 201)
        ),  # Python uses 0-based indexing but will handle internally
    )

    # Run motion compensation
    print("\nRunning motion compensation...")
    compensate_recording(options)
    print("Motion compensation complete!")

    # Read the original video for comparison
    print(f"\nReading original video from {input_file}")
    orig_reader = get_video_file_reader(str(input_file))
    original_frames = orig_reader[:]  # Get all frames
    orig_reader.close()

    # Read the compensated video
    compensated_file = output_folder / "hdf5_comp_minimal" / "compensated.HDF5"
    print(f"Reading compensated video from {compensated_file}")

    vid_reader = get_video_file_reader(str(compensated_file))
    compensated_frames = vid_reader[:]  # Get all frames
    vid_reader.close()

    total_frames = len(compensated_frames)

    # Display videos side-by-side with cv2
    print(
        f"Displaying {total_frames} frames side-by-side. Press 'q' to quit, 'p' to pause/resume"
    )

    # Normalize both videos for display
    def normalize_for_display(frames):
        if frames.dtype != np.uint8:
            frames_min = frames.min()
            frames_max = frames.max()
            if frames_max > frames_min:
                return ((frames - frames_min) / (frames_max - frames_min) * 255).astype(
                    np.uint8
                )
            else:
                return np.zeros_like(frames, dtype=np.uint8)
        return frames

    original_display = normalize_for_display(original_frames)
    compensated_display = normalize_for_display(compensated_frames)

    # Handle multi-channel display
    if original_display.ndim == 4 and original_display.shape[-1] > 1:
        original_display = original_display[..., 0]
    elif original_display.ndim == 4 and original_display.shape[-1] == 1:
        original_display = np.squeeze(original_display, axis=-1)

    if compensated_display.ndim == 4 and compensated_display.shape[-1] > 1:
        compensated_display = compensated_display[..., 0]
    elif compensated_display.ndim == 4 and compensated_display.shape[-1] == 1:
        compensated_display = np.squeeze(compensated_display, axis=-1)

    # Create window
    cv2.namedWindow("Jupiter Demo - Comparison", cv2.WINDOW_NORMAL)

    # Playback settings
    frame_delay = 5
    paused = False
    frame_idx = 0

    while True:
        if not paused:
            # Get current frames
            orig_frame = cv2.cvtColor(original_display[frame_idx], cv2.COLOR_GRAY2BGR)
            comp_frame = cv2.cvtColor(
                compensated_display[frame_idx], cv2.COLOR_GRAY2BGR
            )

            # Add frame number to original (left) image only
            progress_text = f"Frame {frame_idx + 1}/{total_frames}"
            cv2.putText(
                orig_frame,
                progress_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Add labels at bottom of each video
            h, w = orig_frame.shape[:2]
            cv2.putText(
                orig_frame,
                "Uncorrected",
                (w // 2 - 60, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                comp_frame,
                "Corrected",
                (w // 2 - 50, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            # Concatenate frames side by side
            combined_frame = np.hstack([orig_frame, comp_frame])

            # Display combined frame
            cv2.imshow("Jupiter Demo - Comparison", combined_frame)

            # Advance to next frame
            frame_idx = (frame_idx + 1) % total_frames

        # Handle keyboard input
        key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            if paused:
                print("Paused. Press 'p' to resume.")
            else:
                print("Resumed.")
        elif key == ord("r"):
            frame_idx = 0
            print("Restarted from beginning.")

    cv2.destroyAllWindows()
    print("\nPlayback finished.")


if __name__ == "__main__":
    main()
