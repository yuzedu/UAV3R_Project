#!/usr/bin/env python3

import cv2
import os
import argparse
from tqdm import tqdm

def rotate_images(input_dir, output_dir, rotation='cw90'):
    """
    Rotate images in a directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save rotated images
        rotation: Rotation type - 'cw90' (clockwise 90), 'ccw90' (counter-clockwise 90), '180'
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Rotation: {rotation}")
    print(f"Output directory: {output_dir}")

    # Set rotation code
    if rotation == 'cw90':
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotation == 'ccw90':
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    elif rotation == '180':
        rotate_code = cv2.ROTATE_180
    else:
        print(f"Invalid rotation: {rotation}")
        return

    # Process images
    for image_file in tqdm(image_files, desc="Rotating images"):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        # Read image
        img = cv2.imread(input_path)

        if img is None:
            print(f"Warning: Could not read {image_file}")
            continue

        # Rotate image
        rotated_img = cv2.rotate(img, rotate_code)

        # Save rotated image
        cv2.imwrite(output_path, rotated_img)

    print(f"\nSuccessfully rotated {len(image_files)} images!")
    print(f"Rotated images saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rotate images in a directory')
    parser.add_argument('--input', type=str, required=True, help='Input directory with images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for rotated images')
    parser.add_argument('--rotation', type=str, default='cw90', choices=['cw90', 'ccw90', '180'],
                        help='Rotation type: cw90 (clockwise 90°), ccw90 (counter-clockwise 90°), 180 (180°)')

    args = parser.parse_args()

    rotate_images(args.input, args.output, args.rotation)
