#!/usr/bin/env python3

import cv2
import os
import argparse
from tqdm import tqdm

def downsample_images(input_dir, output_dir, height=720, quality=100, format='png'):
    """
    Downsample images to a specific height while maintaining aspect ratio.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save downsampled images
        height: Target height in pixels (default: 720)
        quality: JPEG quality 0-100 or PNG compression 0-9 (default: 100)
        format: Output format 'png' or 'jpg' (default: 'png')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Target height: {height}p")
    print(f"Output format: {format.upper()}")
    print(f"Output directory: {output_dir}")

    # Set up encoding parameters
    if format.lower() == 'png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 0 = max quality
        ext = '.png'
    elif format.lower() == 'jpg' or format.lower() == 'jpeg':
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = '.jpg'
    else:
        encode_params = []
        ext = f'.{format}'

    # Process images
    for image_file in tqdm(image_files, desc="Downsampling images"):
        input_path = os.path.join(input_dir, image_file)

        # Change extension if format changed
        output_filename = os.path.splitext(image_file)[0] + ext
        output_path = os.path.join(output_dir, output_filename)

        # Read image
        img = cv2.imread(input_path)

        if img is None:
            print(f"Warning: Could not read {image_file}")
            continue

        # Get current dimensions
        current_height, current_width = img.shape[:2]

        # Calculate new dimensions maintaining aspect ratio
        if current_height != height:
            aspect_ratio = current_width / current_height
            new_height = height
            new_width = int(height * aspect_ratio)

            # Resize image with high-quality interpolation
            downsampled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            downsampled_img = img

        # Save downsampled image
        cv2.imwrite(output_path, downsampled_img, encode_params)

    print(f"\nSuccessfully downsampled {len(image_files)} images!")
    print(f"Downsampled images saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample images to a specific resolution')
    parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for downsampled images')
    parser.add_argument('--height', type=int, default=720, help='Target height in pixels (default: 720)')
    parser.add_argument('--quality', type=int, default=100, help='JPEG quality 0-100 (default: 100)')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg', 'jpeg'],
                        help='Output image format (default: png)')

    args = parser.parse_args()

    downsample_images(args.input, args.output, args.height, args.quality, args.format)
