#!/usr/bin/env python3
"""
Convert 3D points and camera poses to COLMAP format

COLMAP Binary Format:
- cameras.bin: Camera intrinsics
- images.bin: Camera poses (extrinsics) and 2D keypoints
- points3D.bin: 3D points with color and observations

References: https://colmap.github.io/format.html
"""

import numpy as np
import struct
import argparse
from pathlib import Path


def write_cameras_binary(cameras, output_path):
    """
    Write cameras.bin in COLMAP binary format

    Args:
        cameras: dict {camera_id: (model, width, height, params)}
                 model: 'PINHOLE' or 'SIMPLE_PINHOLE' etc.
                 params: [fx, fy, cx, cy] for PINHOLE
        output_path: path to cameras.bin
    """
    with open(output_path, 'wb') as f:
        # Write number of cameras
        f.write(struct.pack('Q', len(cameras)))

        for camera_id, (model, width, height, params) in cameras.items():
            # Model name to ID mapping
            model_ids = {
                'SIMPLE_PINHOLE': 0,
                'PINHOLE': 1,
                'SIMPLE_RADIAL': 2,
                'RADIAL': 3,
                'OPENCV': 4,
            }
            model_id = model_ids.get(model, 1)

            # Write camera
            f.write(struct.pack('i', camera_id))
            f.write(struct.pack('i', model_id))
            f.write(struct.pack('Q', width))
            f.write(struct.pack('Q', height))
            f.write(struct.pack(f'{len(params)}d', *params))

    print(f"Written {len(cameras)} cameras to {output_path}")


def write_images_binary(images, output_path):
    """
    Write images.bin in COLMAP binary format

    Args:
        images: dict {image_id: (qvec, tvec, camera_id, name, xys, point3D_ids)}
                qvec: quaternion [qw, qx, qy, qz]
                tvec: translation [tx, ty, tz]
                xys: Nx2 array of 2D points
                point3D_ids: N array of corresponding 3D point IDs (-1 if not observed)
        output_path: path to images.bin
    """
    with open(output_path, 'wb') as f:
        # Write number of images
        f.write(struct.pack('Q', len(images)))

        for image_id, (qvec, tvec, camera_id, name, xys, point3D_ids) in images.items():
            # Write image header
            f.write(struct.pack('i', image_id))
            f.write(struct.pack('4d', *qvec))  # qw, qx, qy, qz
            f.write(struct.pack('3d', *tvec))
            f.write(struct.pack('i', camera_id))

            # Write image name (null-terminated string)
            f.write(name.encode('utf-8'))
            f.write(b'\x00')

            # Write 2D points
            num_points2D = len(xys)
            f.write(struct.pack('Q', num_points2D))
            for xy, point3D_id in zip(xys, point3D_ids):
                f.write(struct.pack('2d', *xy))
                f.write(struct.pack('q', point3D_id))

    print(f"Written {len(images)} images to {output_path}")


def write_points3D_binary(points3D, output_path):
    """
    Write points3D.bin in COLMAP binary format

    Args:
        points3D: dict {point3D_id: (xyz, rgb, error, track)}
                  xyz: [x, y, z] coordinates
                  rgb: [r, g, b] color (0-255)
                  error: reprojection error
                  track: list of (image_id, point2D_idx) tuples
        output_path: path to points3D.bin
    """
    with open(output_path, 'wb') as f:
        # Write number of points
        f.write(struct.pack('Q', len(points3D)))

        for point3D_id, (xyz, rgb, error, track) in points3D.items():
            # Write point
            f.write(struct.pack('Q', point3D_id))
            f.write(struct.pack('3d', *xyz))
            f.write(struct.pack('3B', *rgb))
            f.write(struct.pack('d', error))

            # Write track
            f.write(struct.pack('Q', len(track)))
            for image_id, point2D_idx in track:
                f.write(struct.pack('i', image_id))
                f.write(struct.pack('i', point2D_idx))

    print(f"Written {len(points3D)} 3D points to {output_path}")


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion [qw, qx, qy, qz]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz])


def convert_to_colmap(poses, points_3d, feature_tracks, K, image_names,
                      output_dir, image_width=640, image_height=480):
    """
    Convert poses and 3D points to COLMAP format

    Args:
        poses: list of (R, t) tuples for each image
        points_3d: Nx3 array of 3D points
        feature_tracks: dict {point3D_id: [(image_id, (u, v)), ...]}
        K: 3x3 intrinsic matrix
        image_names: list of image filenames
        output_dir: output directory for COLMAP files
        image_width: image width in pixels
        image_height: image height in pixels
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create cameras dict (assuming single camera)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cameras = {
        1: ('PINHOLE', image_width, image_height, [fx, fy, cx, cy])
    }

    # Create images dict
    images = {}
    for img_id, ((R, t), img_name) in enumerate(zip(poses, image_names), start=1):
        # Convert R, t to quaternion
        qvec = rotation_matrix_to_quaternion(R)
        tvec = t.flatten()

        # Find 2D points observed in this image
        xys = []
        point3D_ids = []

        for pt3D_id, observations in feature_tracks.items():
            for obs_img_id, (u, v) in observations:
                if obs_img_id == img_id - 1:  # 0-indexed to 1-indexed
                    xys.append([u, v])
                    point3D_ids.append(pt3D_id)

        if len(xys) == 0:
            xys = np.zeros((0, 2))
            point3D_ids = np.array([], dtype=np.int64)
        else:
            xys = np.array(xys)
            point3D_ids = np.array(point3D_ids, dtype=np.int64)

        images[img_id] = (qvec, tvec, 1, img_name, xys, point3D_ids)

    # Create points3D dict
    points3D_dict = {}
    for pt3D_id, xyz in enumerate(points_3d):
        # Default white color if no color info
        rgb = np.array([255, 255, 255], dtype=np.uint8)
        error = 0.0

        # Build track from feature_tracks
        track = []
        if pt3D_id in feature_tracks:
            for img_id, _ in feature_tracks[pt3D_id]:
                # Find point2D index in that image
                img_data = images[img_id + 1]  # 0-indexed to 1-indexed
                point3D_ids_in_img = img_data[5]

                if pt3D_id in point3D_ids_in_img:
                    point2D_idx = np.where(point3D_ids_in_img == pt3D_id)[0][0]
                    track.append((img_id + 1, point2D_idx))

        points3D_dict[pt3D_id] = (xyz, rgb, error, track)

    # Write binary files
    write_cameras_binary(cameras, output_dir / 'cameras.bin')
    write_images_binary(images, output_dir / 'images.bin')
    write_points3D_binary(points3D_dict, output_dir / 'points3D.bin')

    print(f"\nCOLMAP format conversion complete!")
    print(f"Output directory: {output_dir}")


def main():
    """
    Example usage
    """
    parser = argparse.ArgumentParser(description='Convert poses and 3D points to COLMAP format')
    parser.add_argument('--output', type=str, default='./output/sparse/0',
                      help='Output directory for COLMAP files')
    args = parser.parse_args()

    # Example data (replace with your actual data)
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)

    # Example poses
    import cv2
    poses = [
        (np.eye(3), np.array([0, 0, 0])),
        (cv2.Rodrigues(np.array([0.1, 0, 0]))[0], np.array([1, 0, 0])),
        (cv2.Rodrigues(np.array([0, 0.1, 0]))[0], np.array([0, 1, 0]))
    ]

    # Example 3D points
    points_3d = np.array([
        [1.0, 2.0, 5.0],
        [2.0, 3.0, 6.0],
        [3.0, 1.0, 7.0]
    ])

    # Example feature tracks
    feature_tracks = {
        0: [(0, (320, 240)), (1, (350, 245))],
        1: [(0, (400, 300)), (1, (420, 310)), (2, (410, 305))],
        2: [(1, (250, 200)), (2, (270, 205))]
    }

    image_names = [f'image_{i:04d}.jpg' for i in range(len(poses))]

    convert_to_colmap(poses, points_3d, feature_tracks, K, image_names,
                     args.output, image_width=640, image_height=480)


if __name__ == '__main__':
    main()
