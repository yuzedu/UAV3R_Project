#!/usr/bin/env python3

import cv2
import os
import argparse
from tqdm import tqdm

def extract_images_from_video(video_path, output_dir, sample_rate=1, start_frame=0, end_frame=None, quality=100, format='png'):
    """
    Extract images from a video file with maximum quality.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted images
        sample_rate: Extract every Nth frame (default: 1 for all frames)
        start_frame: Starting frame number (default: 0)
        end_frame: Ending frame number (default: None for all frames)
        quality: JPEG quality 0-100 or PNG compression 0-9 (default: 100 for max quality)
        format: Output format 'png' or 'jpg' (default: 'png')
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video file with backend preference for better quality
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print(f"  Sample rate: every {sample_rate} frame(s)")

    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)

    # Calculate expected output frames
    expected_frames = len(range(start_frame, end_frame, sample_rate))
    print(f"  Expected output: {expected_frames} images")
    print(f"  Output format: {format.upper()}")
    print(f"  Quality setting: {quality}")
    print(f"Extracting images to: {output_dir}")

    # Set up encoding parameters for maximum quality
    if format.lower() == 'png':
        # PNG: 0 = max compression (slower), 9 = min compression (faster, larger files)
        # Use 0 for best quality
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
        ext = '.png'
    elif format.lower() == 'jpg' or format.lower() == 'jpeg':
        # JPEG: 0-100, higher is better quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = '.jpg'
    else:
        encode_params = []
        ext = f'.{format}'

    # Extract frames
    frame_count = 0
    saved_count = 0

    # Set starting position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame

    with tqdm(total=expected_frames, desc="Extracting frames") as pbar:
        while frame_count < end_frame:
            ret, frame = cap.read()

            if not ret:
                break

            # Save frame if it matches the sample rate
            if (frame_count - start_frame) % sample_rate == 0:
                # Create filename with zero-padded frame number
                filename = f"{frame_count:06d}{ext}"
                filepath = os.path.join(output_dir, filename)

                # Write with high quality encoding parameters
                cv2.imwrite(filepath, frame, encode_params)
                saved_count += 1
                pbar.update(1)

            frame_count += 1

    cap.release()
    print(f"\nExtracted {saved_count} images successfully!")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract high-quality images from video file')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for images')
    parser.add_argument('--sample', type=int, default=1,
                        help='Sample rate: extract every Nth frame (default: 1)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting frame number (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='Ending frame number (default: None for all frames)')
    parser.add_argument('--quality', type=int, default=100,
                        help='JPEG quality 0-100 (default: 100 for maximum quality)')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg', 'jpeg'],
                        help='Output image format (default: png for lossless)')

    args = parser.parse_args()

    extract_images_from_video(args.video, args.output, args.sample, args.start, args.end, args.quality, args.format)
