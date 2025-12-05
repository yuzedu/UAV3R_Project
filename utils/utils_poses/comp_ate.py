# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import utils.utils_poses.ATE.trajectory_utils as tu
import utils.utils_poses.ATE.transformations as tf
def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error

def compute_rpe(gt, pred):
    trans_errors = []
    rot_errors = []
    per_view_trans_errors = []
    per_view_rot_errors = []

    for i in range(len(gt) - 1):
        gt1 = gt[i]
        gt2 = gt[i + 1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i + 1]
        pred_rel = np.linalg.inv(pred1) @ pred2

        rel_err = np.linalg.inv(gt_rel) @ pred_rel

        trans_error = translation_error(rel_err)
        rot_error = rotation_error(rel_err)

        trans_errors.append(trans_error)
        rot_errors.append(rot_error)

        per_view_trans_errors.append(trans_error)
        per_view_rot_errors.append(rot_error)

    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))

    return rpe_trans, rpe_rot, per_view_trans_errors, per_view_rot_errors


def compute_ATE(gt, pred):
    """Compute RMSE of ATE
    Args:
        gt: ground-truth poses
        pred: predicted poses
    """
    errors = []
    per_view_ate_errors = []

    for i in range(len(pred)):
        cur_gt = gt[i]
        gt_xyz = cur_gt[:3, 3]

        cur_pred = pred[i]
        pred_xyz = cur_pred[:3, 3]

        align_err = gt_xyz - pred_xyz
        per_frame_error = np.sqrt(np.sum(align_err ** 2))

        errors.append(per_frame_error)
        per_view_ate_errors.append(per_frame_error)

    ate = np.sqrt(np.mean(np.asarray(errors) ** 2))

    return ate, per_view_ate_errors


