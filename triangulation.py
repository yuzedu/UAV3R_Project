#!/usr/bin/env python3
"""
DLT Triangulation with Bundle Adjustment
Inputs:
- Camera poses (rotation + translation) for each image
- 2D feature correspondences across image sequences
"""

import numpy as np
from scipy.optimize import least_squares
import cv2


def triangulate_dlt(P_matrices, points_2d):
    """
    Direct Linear Transform triangulation for multiple views

    Args:
        P_matrices: list of 3x4 projection matrices [P1, P2, ..., Pn]
        points_2d: list of 2D points [(u1,v1), (u2,v2), ..., (un,vn)]

    Returns:
        3D point (X, Y, Z, W) in homogeneous coordinates
    """
    n_views = len(P_matrices)
    A = np.zeros((2 * n_views, 4))

    for i, (P, pt) in enumerate(zip(P_matrices, points_2d)):
        u, v = pt
        A[2*i] = u * P[2] - P[0]
        A[2*i + 1] = v * P[2] - P[1]

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]

    # Convert to 3D
    return X[:3] / X[3]


def projection_matrix(R, t, K):
    """
    Construct projection matrix P = K[R|t]

    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        K: 3x3 camera intrinsic matrix

    Returns:
        3x4 projection matrix
    """
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def reproject(point_3d, R, t, K):
    """
    Reproject 3D point to 2D using camera parameters

    Args:
        point_3d: 3D point
        R: rotation matrix
        t: translation vector
        K: intrinsic matrix

    Returns:
        2D reprojected point (u, v)
    """
    P = projection_matrix(R, t, K)
    point_homo = np.append(point_3d, 1)
    projected = P @ point_homo
    return projected[:2] / projected[2]


def reprojection_error_points_only(points_3d, observations, camera_indices, point_indices, poses, K):
    """
    Compute reprojection error for bundle adjustment (3D points only, poses fixed)

    Args:
        points_3d: flattened array of 3D points to optimize
        observations: 2D observations
        camera_indices: which camera for each observation
        point_indices: which 3D point for each observation
        poses: list of fixed (R, t) camera poses
        K: intrinsic matrix

    Returns:
        residuals: reprojection errors
    """
    n_points = len(points_3d) // 3
    points_3d = points_3d.reshape((n_points, 3))

    residuals = []

    for obs, cam_idx, pt_idx in zip(observations, camera_indices, point_indices):
        # Get fixed camera pose
        R, t = poses[cam_idx]

        # Reproject
        projected = reproject(points_3d[pt_idx], R, t, K)

        # Compute error
        residuals.append(obs[0] - projected[0])
        residuals.append(obs[1] - projected[1])

    return np.array(residuals)


def bundle_adjustment(poses, points_2d_all, points_3d_init, K, fix_poses=True):
    """
    Bundle adjustment to refine 3D points (poses assumed correct and fixed)

    Args:
        poses: list of (R, t) tuples for each camera (FIXED, not optimized)
        points_2d_all: dictionary mapping point_id to list of (camera_idx, (u,v)) observations
        points_3d_init: initial 3D point estimates
        K: camera intrinsic matrix
        fix_poses: if True, only optimize 3D points (default: True)

    Returns:
        poses: original poses (unchanged)
        optimized_points: refined 3D points
    """
    n_points = len(points_3d_init)
    points_params = np.array(points_3d_init)

    # Flatten 3D points for optimization
    x0 = points_params.flatten()

    # Prepare observations
    observations = []
    camera_indices = []
    point_indices = []

    for pt_idx, obs_list in points_2d_all.items():
        for cam_idx, pt_2d in obs_list:
            observations.append(pt_2d)
            camera_indices.append(cam_idx)
            point_indices.append(pt_idx)

    observations = np.array(observations)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)

    print(f"Optimizing {n_points} 3D points with {len(observations)} observations")
    print(f"Camera poses are FIXED (assumed correct)")

    # Run optimization - only optimize 3D points
    result = least_squares(
        reprojection_error_points_only,
        x0,
        args=(observations, camera_indices, point_indices, poses, K),
        verbose=2,
        max_nfev=200,
        ftol=1e-6,
        xtol=1e-6
    )

    # Extract optimized 3D points
    opt_points = result.x.reshape((n_points, 3))

    # Return original poses (unchanged) and optimized points
    return poses, opt_points


def triangulate_points(poses, feature_tracks, K):
    """
    Triangulate multiple 3D points from feature tracks across images

    Args:
        poses: list of (R, t) for each camera/image
        feature_tracks: dictionary mapping track_id to list of (image_idx, (u,v))
        K: 3x3 camera intrinsic matrix

    Returns:
        points_3d: dictionary mapping track_id to 3D point
    """
    points_3d = {}

    for track_id, observations in feature_tracks.items():
        if len(observations) < 2:
            continue

        # Get projection matrices and 2D points for this track
        P_matrices = []
        points_2d = []

        for img_idx, pt_2d in observations:
            R, t = poses[img_idx]
            P = projection_matrix(R, t, K)
            P_matrices.append(P)
            points_2d.append(pt_2d)

        # Triangulate
        point_3d = triangulate_dlt(P_matrices, points_2d)
        points_3d[track_id] = point_3d

    return points_3d


def main():
    """
    Example usage
    """
    # Example: 3 cameras observing 5 points
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)

    # Example poses (identity for first camera, then slight rotations/translations)
    poses = [
        (np.eye(3), np.array([0, 0, 0])),
        (cv2.Rodrigues(np.array([0.1, 0, 0]))[0], np.array([1, 0, 0])),
        (cv2.Rodrigues(np.array([0, 0.1, 0]))[0], np.array([0, 1, 0]))
    ]

    # Example feature tracks
    # Format: {track_id: [(image_idx, (u, v)), ...]}
    feature_tracks = {
        0: [(0, (320, 240)), (1, (350, 245)), (2, (330, 250))],
        1: [(0, (400, 300)), (1, (420, 310)), (2, (410, 305))],
        2: [(0, (250, 200)), (1, (270, 205))],
    }

    # Initial triangulation
    print("Performing initial triangulation...")
    points_3d_init = triangulate_points(poses, feature_tracks, K)

    print(f"Triangulated {len(points_3d_init)} points")
    for track_id, pt in points_3d_init.items():
        print(f"Track {track_id}: {pt}")

    # Prepare data for bundle adjustment
    points_2d_all = feature_tracks
    points_3d_list = [points_3d_init[i] for i in sorted(points_3d_init.keys())]

    # Bundle adjustment
    print("\nPerforming bundle adjustment...")
    opt_poses, opt_points = bundle_adjustment(poses, points_2d_all, points_3d_list, K)

    print("\nOptimized 3D points:")
    for i, pt in enumerate(opt_points):
        print(f"Point {i}: {pt}")


if __name__ == "__main__":
    main()
