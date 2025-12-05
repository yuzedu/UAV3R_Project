#!/usr/bin/env python3

import cv2
import os
from PIL import Image
from tqdm import tqdm
import argparse

def video_to_gif(video_path, output_path, scale=1.0, fps_reduction=1, quality=100):
    """
    Convert video to high-quality GIF.

    Args:
        video_path: Path to input video
        output_path: Path to output GIF
        scale: Scale factor (1.0 = original size, 0.5 = half size, etc.)
        fps_reduction: Keep every Nth frame (1=all frames, 2=half fps, etc.)
        quality: GIF quality/colors (1-100, higher is better but larger file)
    """
    print(f"Converting {video_path} to GIF...")

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Calculate GIF FPS
    gif_fps = fps / fps_reduction
    frame_duration = int(1000 / gif_fps)  # milliseconds per frame

    print(f"Original: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    print(f"Output: {new_width}x{new_height} @ {gif_fps:.2f} FPS")
    print(f"Extracting every {fps_reduction} frame(s)...")

    # Extract frames
    frames = []
    frame_count = 0

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Keep every Nth frame
            if frame_count % fps_reduction == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize if needed
                if scale != 1.0:
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height),
                                          interpolation=cv2.INTER_LANCZOS4)

                # Convert to PIL Image
                pil_img = Image.fromarray(frame_rgb)
                frames.append(pil_img)

            frame_count += 1
            pbar.update(1)

    cap.release()

    print(f"Extracted {len(frames)} frames")
    print(f"Creating GIF with {frame_duration}ms per frame...")

    # Save as GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True,
            quality=quality
        )

        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nGIF created successfully!")
        print(f"Output: {output_path}")
        print(f"Size: {file_size_mb:.2f} MB")
        print(f"Resolution: {new_width}x{new_height}")
        print(f"Frames: {len(frames)}")
        print(f"Duration: {len(frames) * frame_duration / 1000:.2f}s")
    else:
        print("Error: No frames extracted!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert video to high-quality GIF')
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--output', type=str, required=True, help='Output GIF file')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor (1.0=original, 0.5=half, 0.375=1440p from 4K)')
    parser.add_argument('--fps-reduction', type=int, default=2,
                       help='Keep every Nth frame (1=all, 2=half fps, etc.)')
    parser.add_argument('--quality', type=int, default=85,
                       help='GIF quality 1-100 (default: 85)')

    args = parser.parse_args()

    video_to_gif(args.video, args.output, args.scale, args.fps_reduction, args.quality)
