"""
Jupiter Demo - Motion Compensation Example
Author: Philipp Flotho (Python port)
Copyright 2021 by Philipp Flotho, All rights reserved.

This example downloads jupiter demo data and demonstrates minimal motion compensation config.
"""

import os
from pathlib import Path
import urllib.request
import numpy as np
import cv2
from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.motion_correction.compensate_recording import compensate_recording
from pyflowreg.util.io.factory import get_video_file_reader


def download_jupiter_data(output_folder: Path, input_file: Path) -> None:
    """Download jupiter demo data if not already present."""
    if not input_file.exists():
        print("Downloading jupiter demo data...")
        # Google Drive download link for jupiter.tiff
        url = "https://drive.usercontent.google.com/download?id=12lEhahzKtOZsFgxLzwxnT8JsVBErvzJH&export=download&authuser=0"
        urllib.request.urlretrieve(url, input_file)
        print(f"Downloaded to {input_file}")
    else:
        print(f"Jupiter data already exists at {input_file}")


def main():
    # Prepare output directory
    output_folder = Path("jupiter_demo")
    output_folder.mkdir(exist_ok=True)
    
    # Input file path
    input_file = output_folder / "jupiter.tiff"
    
    # Download data if needed
    download_jupiter_data(output_folder, input_file)
    
    # Create OF_options matching MATLAB configuration
    options = OFOptions(
        input_file=str(input_file),
        output_path=str(output_folder / "hdf5_comp_minimal"),
        output_format="HDF5",
        alpha=4,  # Larger alpha to avoid registering changing morphology
        quality_setting="balanced",  # Default in MATLAB is 'quality'
        output_typename="",
        reference_frames=list(range(100, 201))  # Python uses 0-based indexing but will handle internally
    )
    
    # Run motion compensation
    print("\nRunning motion compensation...")
    compensate_recording(options)
    print("Motion compensation complete!")
    
    # Read the compensated video
    compensated_file = output_folder / "hdf5_comp_minimal" / "compensated.HDF5"
    print(f"\nReading compensated video from {compensated_file}")
    
    vid_reader = get_video_file_reader(str(compensated_file))
    total_frames = len(vid_reader)
    
    # Display video with cv2
    print(f"Displaying {total_frames} frames. Press 'q' to quit, 'p' to pause/resume")
    
    # Read all frames using array-like indexing
    frames = vid_reader[:]  # Get all frames
    
    # Normalize frames for display if needed
    if frames.dtype != np.uint8:
        # Convert to uint8 for display
        frames_min = frames.min()
        frames_max = frames.max()
        if frames_max > frames_min:
            frames = ((frames - frames_min) / (frames_max - frames_min) * 255).astype(np.uint8)
        else:
            frames = np.zeros_like(frames, dtype=np.uint8)
    
    # Create window
    cv2.namedWindow('Jupiter Demo - Compensated', cv2.WINDOW_NORMAL)
    
    # Playback settings
    frame_delay = 1
    paused = False
    frame_idx = 0
    
    while True:
        if not paused:

            frame = cv2.cvtColor(frames[frame_idx], cv2.COLOR_GRAY2BGR)

            # Add progress counter
            progress_text = f"Frame {frame_idx + 1}/{total_frames} ({100 * (frame_idx + 1) / total_frames:.1f}%)"
            
            # Display frame
            cv2.imshow('Jupiter Demo - Compensated', frame)
            
            # Advance to next frame
            frame_idx = (frame_idx + 1) % total_frames
        
        # Handle keyboard input
        key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Paused. Press 'p' to resume.")
            else:
                print("Resumed.")
        elif key == ord('r'):
            frame_idx = 0
            print("Restarted from beginning.")
    
    cv2.destroyAllWindows()
    print("\nPlayback finished.")


if __name__ == "__main__":
    main()