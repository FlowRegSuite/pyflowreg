"""
Jupiter Demo Live - Online motion compensation with FlowRegLive.

Demonstrates real-time motion compensation using FlowRegLive class
with framerate display for performance monitoring.
"""

import time
import numpy as np
import cv2
from pyflowreg.motion_correction.OF_options import OFOptions, QualitySetting
from pyflowreg.motion_correction.flow_reg_live import FlowRegLive
from pyflowreg.util.io.factory import get_video_file_reader
from pyflowreg.util.download import download_demo_data


def main():
    # Download data to data/ folder (default location)
    input_file = download_demo_data("jupiter.tiff")

    # Read the entire video into memory
    print("\nReading jupiter video into memory...")
    reader = get_video_file_reader(str(input_file))

    # Get video properties
    print(f"Video shape: {reader.shape}")
    print(f"Video dtype: {reader.dtype}")

    # Read all frames into array
    video_array = reader[:]  # Read all frames
    print(f"Loaded video array shape: {video_array.shape}, dtype: {video_array.dtype}")
    reader.close()

    # Create OF_options for FlowRegLive (fast settings)
    options = OFOptions(
        alpha=4,  # Larger alpha to avoid registering changing morphology
        quality_setting=QualitySetting.FAST,  # Will be overridden to FAST anyway
        sigma=[[2.0, 2.0, 0.5], [2.0, 2.0, 0.5]],  # Gaussian filtering parameters
        levels=100,
        iterations=50,
        eta=0.8,
        channel_normalization="separate",  # Per-channel normalization
    )

    # Initialize FlowRegLive
    print("\nInitializing FlowRegLive...")
    flow_reg = FlowRegLive(
        options=options,
        reference_buffer_size=50,  # Use first 50 frames for reference
        reference_update_interval=20,  # Update reference every 20 frames
        reference_update_weight=0.2,  # Mix in 20% of new frame
    )

    # Set reference from frames 100-200 (matching original demo)
    print("Setting reference from frames 100-200...")
    reference_frames = video_array[100:201]
    flow_reg.set_reference(reference_frames)

    # Prepare display
    print("\nStarting live motion compensation demo...")
    print("Press 'q' to quit, 'p' to pause/resume, 'r' to restart")
    print("Displaying side-by-side: original (left) vs corrected (right)")

    # Create window
    cv2.namedWindow("Jupiter Live Demo - Real-time Compensation", cv2.WINDOW_NORMAL)

    # Playback settings
    paused = False
    frame_idx = 0
    total_frames = len(video_array)

    # FPS calculation
    fps_buffer = []
    fps_buffer_size = 30  # Average over last 30 frames
    last_time = time.time()

    # Main loop - endless loop through video
    while True:
        if not paused:
            current_time = time.time()

            # Get current frame
            frame = video_array[frame_idx]

            # Process with FlowRegLive
            start_process = time.time()
            registered, flow = flow_reg(frame)
            process_time = time.time() - start_process

            # Calculate FPS
            frame_time = current_time - last_time
            if frame_time > 0:
                fps = 1.0 / frame_time
                fps_buffer.append(fps)
                if len(fps_buffer) > fps_buffer_size:
                    fps_buffer.pop(0)
            last_time = current_time

            avg_fps = np.mean(fps_buffer) if fps_buffer else 0

            # Prepare frames for display
            def normalize_for_display(arr):
                if arr.dtype != np.uint8:
                    arr_min = arr.min()
                    arr_max = arr.max()
                    if arr_max > arr_min:
                        return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(
                            np.uint8
                        )
                    else:
                        return np.zeros_like(arr, dtype=np.uint8)
                return arr

            # Handle multi-channel
            if frame.ndim == 3 and frame.shape[-1] > 1:
                frame_display = frame[..., 0]
            elif frame.ndim == 3 and frame.shape[-1] == 1:
                frame_display = np.squeeze(frame, axis=-1)
            else:
                frame_display = frame

            if registered.ndim == 3 and registered.shape[-1] > 1:
                registered_display = registered[..., 0]
            elif registered.ndim == 3 and registered.shape[-1] == 1:
                registered_display = np.squeeze(registered, axis=-1)
            else:
                registered_display = registered

            # Normalize for display
            frame_display = normalize_for_display(frame_display)
            registered_display = normalize_for_display(registered_display)

            # Convert to BGR for display
            orig_frame = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
            reg_frame = cv2.cvtColor(registered_display, cv2.COLOR_GRAY2BGR)

            # Add frame info to original (left)
            info_text = f"Frame {frame_idx + 1}/{total_frames}"
            cv2.putText(
                orig_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Add FPS and processing time info
            fps_text = f"FPS: {avg_fps:.1f} | Process: {process_time*1000:.1f}ms"
            cv2.putText(
                orig_frame,
                fps_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Add flow magnitude info
            flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_text = f"Flow: max={flow_mag.max():.2f}, mean={flow_mag.mean():.2f}"
            cv2.putText(
                orig_frame,
                flow_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # Add labels at bottom
            h, w = orig_frame.shape[:2]
            cv2.putText(
                orig_frame,
                "Original",
                (w // 2 - 40, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                reg_frame,
                "Corrected (Live)",
                (w // 2 - 70, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            # Reference update indicator
            if flow_reg.frame_count % flow_reg.reference_update_interval == 0:
                cv2.putText(
                    reg_frame,
                    "REF UPDATE",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Concatenate frames side by side
            combined_frame = np.hstack([orig_frame, reg_frame])

            # Display combined frame
            cv2.imshow("Jupiter Live Demo - Real-time Compensation", combined_frame)

            # Advance to next frame (loop back to start)
            frame_idx = (frame_idx + 1) % total_frames

            # Show message when looping
            if frame_idx == 0:
                print(f"Looping back to start... Avg FPS: {avg_fps:.1f}")

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            if paused:
                print("Paused. Press 'p' to resume.")
            else:
                print("Resumed.")
                last_time = time.time()  # Reset timer after pause
        elif key == ord("r"):
            frame_idx = 0
            fps_buffer.clear()
            flow_reg.frame_count = 0
            print("Restarted from beginning.")

    cv2.destroyAllWindows()
    print("\nLive demo finished.")

    # Print statistics
    if fps_buffer:
        print(f"Average FPS: {np.mean(fps_buffer):.1f}")
        print(f"Min FPS: {np.min(fps_buffer):.1f}")
        print(f"Max FPS: {np.max(fps_buffer):.1f}")


if __name__ == "__main__":
    main()
