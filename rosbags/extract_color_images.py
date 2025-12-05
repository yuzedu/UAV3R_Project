#!/usr/bin/env python3

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import sqlite3
import cv2
from cv_bridge import CvBridge
import os
from tqdm import tqdm

def extract_images_from_bag(bag_path, output_dir, topic_name='/camera/camera/color/image_raw', sample_rate=1):
    """
    Extract images from ROS2 bag file.

    Args:
        bag_path: Path to the ROS2 bag directory (containing .db3 file)
        output_dir: Directory to save extracted images
        topic_name: Topic name to extract images from
        sample_rate: Extract every Nth message (default: 1)
    """
    # Find the .db3 file
    db3_files = [f for f in os.listdir(bag_path) if f.endswith('.db3')]
    if not db3_files:
        print(f"No .db3 file found in {bag_path}")
        return

    db_path = os.path.join(bag_path, db3_files[0])
    print(f"Reading bag from: {db_path}")

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get topic id for the color image topic
    cursor.execute("SELECT id FROM topics WHERE name=?", (topic_name,))
    topic_result = cursor.fetchone()

    if not topic_result:
        print(f"Topic {topic_name} not found in bag!")
        print("Available topics:")
        cursor.execute("SELECT name FROM topics")
        for row in cursor.fetchall():
            print(f"  - {row[0]}")
        conn.close()
        return

    topic_id = topic_result[0]

    # Get message type
    cursor.execute("SELECT type FROM topics WHERE id=?", (topic_id,))
    msg_type = cursor.fetchone()[0]
    print(f"Message type: {msg_type}")

    # Get total message count
    cursor.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,))
    total_messages = cursor.fetchone()[0]
    print(f"Total messages: {total_messages}")
    print(f"Sample rate: every {sample_rate} message(s)")
    print(f"Expected output: {total_messages // sample_rate} images")

    # Get all messages for this topic
    cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize CV Bridge
    bridge = CvBridge()

    # Get message class
    msg_class = get_message(msg_type)

    # Extract images
    print(f"Extracting images to: {output_dir}")
    image_count = 0
    saved_count = 0

    for timestamp, data in tqdm(cursor.fetchall(), total=total_messages, desc="Extracting images"):
        # Sample every Nth message
        if image_count % sample_rate != 0:
            image_count += 1
            continue
        # Deserialize message
        msg = deserialize_message(data, msg_class)

        # Convert ROS image to OpenCV image
        if 'CompressedImage' in msg_type:
            # Handle compressed images
            cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        else:
            # Handle uncompressed images
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Save image with frame number as filename
        filename = f"{saved_count:06d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cv_image)

        saved_count += 1
        image_count += 1

    conn.close()
    print(f"\nExtracted {saved_count} images successfully!")
    print(f"Images saved to: {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract color images from ROS2 bag')
    parser.add_argument('--bag', type=str, required=True, help='Path to ROS2 bag directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for images')
    parser.add_argument('--topic', type=str, default='/camera/camera/color/image_raw',
                        help='Topic name to extract (default: /camera/camera/color/image_raw)')
    parser.add_argument('--sample', type=int, default=1, help='Sample rate: extract every Nth message (default: 1)')

    args = parser.parse_args()

    extract_images_from_bag(args.bag, args.output, args.topic, args.sample)
