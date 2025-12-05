#!/usr/bin/env python3
"""
Extract colored images from camera topic in ROS2 bag

Supports both sensor_msgs/Image and sensor_msgs/CompressedImage
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

try:
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr
    from rosbags.image import message_to_cvimage
except ImportError:
    print("Error: rosbags library not found. Install with:")
    print("  pip install rosbags")
    exit(1)


def extract_images_from_bag(bag_path, camera_topic, output_dir, format='jpg',
                            max_images=None, skip_frames=1):
    """
    Extract images from ROS2 bag file

    Args:
        bag_path: path to ROS2 bag directory
        camera_topic: camera topic name (e.g., '/camera/image_raw')
        output_dir: output directory for extracted images
        format: image format ('jpg', 'png')
        max_images: maximum number of images to extract (None for all)
        skip_frames: extract every Nth frame (1 for all frames)
    """
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening bag: {bag_path}")
    print(f"Camera topic: {camera_topic}")
    print(f"Output directory: {output_dir}")
    print(f"Format: {format}")

    image_count = 0
    frame_count = 0
    timestamps = []

    with Reader(bag_path) as reader:
        # Check available topics
        topics = {conn.topic: conn.msgtype for conn in reader.connections}

        if camera_topic not in topics:
            print(f"\nError: Topic '{camera_topic}' not found in bag!")
            print("Available topics:")
            for topic, msgtype in topics.items():
                print(f"  {topic} ({msgtype})")
            return

        print(f"Topic type: {topics[camera_topic]}")

        # Filter messages from camera topic
        connections = [x for x in reader.connections if x.topic == camera_topic]

        # Count total messages
        total_messages = sum(1 for _ in reader.messages(connections=connections))
        print(f"Total messages in topic: {total_messages}")

        # Extract images
        print("\nExtracting images...")
        for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections),
                                                   total=total_messages):
            # Skip frames if needed
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            # Deserialize message
            msg = deserialize_cdr(rawdata, connection.msgtype)

            # Convert to OpenCV image
            try:
                if 'CompressedImage' in connection.msgtype:
                    # Compressed image
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                else:
                    # Raw image
                    img = message_to_cvimage(msg)

                    # Convert to BGR if needed
                    if msg.encoding == 'rgb8':
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif msg.encoding == 'mono8' or msg.encoding == 'mono16':
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            except Exception as e:
                print(f"\nWarning: Failed to decode image at frame {frame_count}: {e}")
                frame_count += 1
                continue

            # Save image
            output_file = output_dir / f"frame_{image_count:06d}.{format}"
            cv2.imwrite(str(output_file), img)

            # Store timestamp (nanoseconds to seconds)
            timestamps.append(timestamp / 1e9)

            image_count += 1
            frame_count += 1

            # Check max images limit
            if max_images is not None and image_count >= max_images:
                break

    # Save timestamps
    timestamps_file = output_dir / 'timestamps.txt'
    with open(timestamps_file, 'w') as f:
        for i, ts in enumerate(timestamps):
            f.write(f"{i:06d} {ts:.9f}\n")

    print(f"\n✓ Extracted {image_count} images to {output_dir}")
    print(f"✓ Saved timestamps to {timestamps_file}")

    return image_count, timestamps


def main():
    parser = argparse.ArgumentParser(
        description='Extract colored images from ROS2 bag camera topic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all images from camera topic
  python extract_images_from_rosbag.py /path/to/rosbag2 /camera/image_raw -o ./images

  # Extract every 10th frame as PNG
  python extract_images_from_rosbag.py /path/to/rosbag2 /camera/image_raw -o ./images -f png -s 10

  # Extract max 100 images
  python extract_images_from_rosbag.py /path/to/rosbag2 /camera/image_raw -o ./images -m 100
        """
    )

    parser.add_argument('bag_path', type=str,
                       help='Path to ROS2 bag directory')
    parser.add_argument('camera_topic', type=str,
                       help='Camera topic name (e.g., /camera/image_raw)')
    parser.add_argument('-o', '--output', type=str, default='./extracted_images',
                       help='Output directory (default: ./extracted_images)')
    parser.add_argument('-f', '--format', type=str, default='jpg',
                       choices=['jpg', 'png'],
                       help='Image format (default: jpg)')
    parser.add_argument('-m', '--max-images', type=int, default=None,
                       help='Maximum number of images to extract')
    parser.add_argument('-s', '--skip-frames', type=int, default=1,
                       help='Extract every Nth frame (default: 1 = all frames)')
    parser.add_argument('-l', '--list-topics', action='store_true',
                       help='List all available topics and exit')

    args = parser.parse_args()

    # List topics if requested
    if args.list_topics:
        print(f"Listing topics in: {args.bag_path}\n")
        with Reader(Path(args.bag_path)) as reader:
            print(f"{'Topic':<50} {'Type':<50} {'Count':<10}")
            print("-" * 110)

            topic_counts = {}
            for connection, timestamp, rawdata in reader.messages():
                topic_counts[connection.topic] = topic_counts.get(connection.topic, 0) + 1

            topics = {conn.topic: conn.msgtype for conn in reader.connections}
            for topic, msgtype in sorted(topics.items()):
                count = topic_counts.get(topic, 0)
                print(f"{topic:<50} {msgtype:<50} {count:<10}")
        return

    # Extract images
    extract_images_from_bag(
        args.bag_path,
        args.camera_topic,
        args.output,
        format=args.format,
        max_images=args.max_images,
        skip_frames=args.skip_frames
    )


if __name__ == '__main__':
    main()
