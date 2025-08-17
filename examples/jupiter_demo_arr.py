"""
Jupiter Demo Array Version - Test array-based motion compensation
Uses the same jupiter data but processes through compensate_arr instead of files.
"""

import os
from pathlib import Path
import urllib.request
import numpy as np
import cv2
from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.motion_correction.compensate_arr import compensate_arr
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
    
    # Read the entire video into memory using the factory
    print("\nReading jupiter video into memory...")
    reader = get_video_file_reader(str(input_file))
    
    # Get video properties
    print(f"Video shape: {reader.shape}")
    print(f"Video dtype: {reader.dtype}")
    
    # Read all frames into array
    video_array = reader[:]  # Read all frames
    print(f"Loaded video array shape: {video_array.shape}, dtype: {video_array.dtype}")
    
    # Create reference from frames 100-200 (0-based indexing in Python)
    reference_frames = video_array[100:201]
    reference = np.mean(reference_frames, axis=0)
    print(f"Reference shape: {reference.shape}, dtype: {reference.dtype}")
    
    # Close the reader
    reader.close()
    
    # Create OF_options matching the original demo
    options = OFOptions(
        alpha=4,  # Larger alpha to avoid registering changing morphology
        quality_setting="balanced",
        levels=100,  # Default
        iterations=50,  # Default
        eta=0.8,  # Default
        save_w=True,  # Save displacement fields
        output_typename="double"  # Keep double precision
    )
    
    # Run array-based motion compensation
    print("\nRunning array-based motion compensation...")
    print("This uses compensate_arr instead of file-based processing...")
    
    try:
        registered, flow = compensate_arr(video_array, reference, options)
        
        print("\nMotion compensation complete!")
        print(f"Registered shape: {registered.shape}, dtype: {registered.dtype}")
        print(f"Flow fields shape: {flow.shape}, dtype: {flow.dtype}")
        
        # Compute some statistics
        print("\nStatistics:")
        print(f"Original mean: {np.mean(video_array):.6f}")
        print(f"Registered mean: {np.mean(registered):.6f}")
        print(f"Max displacement magnitude: {np.max(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)):.3f} pixels")
        print(f"Mean displacement magnitude: {np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)):.3f} pixels")
        
        # Display video with cv2
        print(f"\nDisplaying registered video. Press 'q' to quit, 'p' to pause/resume")
        
        # Normalize frames for display if needed
        if registered.dtype != np.uint8:
            # Convert to uint8 for display
            frames_min = registered.min()
            frames_max = registered.max()
            if frames_max > frames_min:
                display_frames = ((registered - frames_min) / (frames_max - frames_min) * 255).astype(np.uint8)
            else:
                display_frames = np.zeros_like(registered, dtype=np.uint8)
        else:
            display_frames = registered
        
        # Handle multi-channel display
        if display_frames.ndim == 4 and display_frames.shape[-1] > 1:
            # Take first channel for display
            display_frames = display_frames[..., 0]
        elif display_frames.ndim == 4 and display_frames.shape[-1] == 1:
            # Squeeze single channel
            display_frames = np.squeeze(display_frames, axis=-1)
        
        # Create window
        cv2.namedWindow('Jupiter Demo Array - Registered', cv2.WINDOW_NORMAL)
        
        # Playback settings
        frame_delay = 1
        paused = False
        frame_idx = 0
        total_frames = len(display_frames)
        
        while True:
            if not paused:
                frame = cv2.cvtColor(display_frames[frame_idx], cv2.COLOR_GRAY2BGR)
                
                # Add progress text
                progress_text = f"Frame {frame_idx + 1}/{total_frames} ({100 * (frame_idx + 1) / total_frames:.1f}%)"
                cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Jupiter Demo Array - Registered', frame)
                
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
        
    except Exception as e:
        print(f"\nError during array compensation: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to provide debugging info
        print("\nDebugging info:")
        print(f"Video array shape: {video_array.shape}")
        print(f"Video array dtype: {video_array.dtype}")
        print(f"Reference shape: {reference.shape}")
        print(f"Reference dtype: {reference.dtype}")
        print(f"Video array C-contiguous: {video_array.flags['C_CONTIGUOUS']}")
        print(f"Video array owns data: {video_array.flags['OWNDATA']}")


if __name__ == "__main__":
    main()