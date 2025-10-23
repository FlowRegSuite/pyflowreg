"""
Jupiter Demo Array GPU Version - Test GPU-accelerated motion compensation
Uses PyTorch backend for GPU acceleration with array-based processing.

Note: CuPy backend (flowreg_cuda) is also available on Linux/Windows with CUDA.
"""

import numpy as np
import cv2
import time
from pyflowreg.motion_correction.OF_options import OFOptions
from pyflowreg.motion_correction.compensate_arr import compensate_arr
from pyflowreg.util.io.factory import get_video_file_reader
from pyflowreg.util.download import download_demo_data


def main():
    # Download data to data/ folder (default location)
    input_file = download_demo_data("jupiter.tiff")

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

    # Create OF_options with GPU backend
    # flowreg_torch: PyTorch backend (works on CPU or GPU, all platforms)
    # flowreg_cuda: CuPy backend (GPU only, Linux/Windows with CUDA)
    options = OFOptions(
        alpha=4,  # Larger alpha to avoid registering changing morphology
        quality_setting="balanced",
        levels=100,  # Default
        iterations=50,  # Default
        eta=0.8,  # Default
        save_w=True,  # Save displacement fields
        output_typename="double",  # Keep double precision
        flow_backend="flowreg_torch",  # Use PyTorch GPU backend
        backend_params={
            "device": "cuda",  # Use GPU if available, falls back to CPU
            "dtype": "float32",  # Use float32 for GPU (faster than float64)
        },
    )

    # Run array-based motion compensation with GPU
    print("\nRunning GPU-accelerated array-based motion compensation...")
    print("Backend: PyTorch (flowreg_torch)")
    print("This uses compensate_arr with GPU acceleration...")

    try:
        start_time = time.time()
        registered, flow = compensate_arr(video_array, reference, options)
        elapsed_time = time.time() - start_time

        print("\nMotion compensation complete!")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Registered shape: {registered.shape}, dtype: {registered.dtype}")
        print(f"Flow fields shape: {flow.shape}, dtype: {flow.dtype}")

        # Compute some statistics
        print("\nStatistics:")
        print(f"Original mean: {np.mean(video_array):.6f}")
        print(f"Registered mean: {np.mean(registered):.6f}")
        print(
            f"Max displacement magnitude: {np.max(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)):.3f} pixels"
        )
        print(
            f"Mean displacement magnitude: {np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)):.3f} pixels"
        )

        # Display videos side-by-side with cv2
        print(
            "\nDisplaying side-by-side comparison. Press 'q' to quit, 'p' to pause/resume"
        )

        # Normalize both videos for display
        def normalize_for_display(frames):
            if frames.dtype != np.uint8:
                frames_min = frames.min()
                frames_max = frames.max()
                if frames_max > frames_min:
                    return (
                        (frames - frames_min) / (frames_max - frames_min) * 255
                    ).astype(np.uint8)
                else:
                    return np.zeros_like(frames, dtype=np.uint8)
            return frames

        original_display = normalize_for_display(video_array)
        registered_display = normalize_for_display(registered)

        # Handle multi-channel display
        if original_display.ndim == 4 and original_display.shape[-1] > 1:
            original_display = original_display[..., 0]
        elif original_display.ndim == 4 and original_display.shape[-1] == 1:
            original_display = np.squeeze(original_display, axis=-1)

        if registered_display.ndim == 4 and registered_display.shape[-1] > 1:
            registered_display = registered_display[..., 0]
        elif registered_display.ndim == 4 and registered_display.shape[-1] == 1:
            registered_display = np.squeeze(registered_display, axis=-1)

        # Create window
        cv2.namedWindow("Jupiter Demo GPU - Comparison", cv2.WINDOW_NORMAL)

        # Playback settings
        frame_delay = 5
        paused = False
        frame_idx = 0
        total_frames = len(registered_display)

        while True:
            if not paused:
                # Get current frames
                orig_frame = cv2.cvtColor(
                    original_display[frame_idx], cv2.COLOR_GRAY2BGR
                )
                reg_frame = cv2.cvtColor(
                    registered_display[frame_idx], cv2.COLOR_GRAY2BGR
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
                    reg_frame,
                    "Corrected (GPU)",
                    (w // 2 - 70, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                # Concatenate frames side by side
                combined_frame = np.hstack([orig_frame, reg_frame])

                # Display combined frame
                cv2.imshow("Jupiter Demo GPU - Comparison", combined_frame)

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

    except Exception as e:
        print(f"\nError during GPU array compensation: {e}")
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
