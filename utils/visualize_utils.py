# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
import numpy as np
import copy
import os
import torch
import scipy
import csv
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = 'Agg'
from evo.core.trajectory import PosePath3D
from evo.tools import plot
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE
from utils.graphics_utils import getWorld2View_np


def vis_pose(cameras):
    trj_est_np, trj_gt_np = [], []
    
    for camera in cameras:
        pose_est = np.linalg.inv(getWorld2View_np(camera.R, camera.T))
        R,t = pose_est[:3,:3], pose_est[:3,3]
        rot = Rotation.from_matrix(R)
        quad = rot.as_quat() # xyzw
        quad_wxyz = np.array([quad[3], quad[0], quad[1], quad[2]])
        pose = np.concatenate([t, quad_wxyz], 0)
        trj_est_np.append(pose_est)
    trj_est_np = np.stack(trj_est_np, 0)
        
    for camera in cameras:
        pose_gt = np.linalg.inv(getWorld2View_np(camera.R_gt, camera.T_gt))
        R,t = pose_gt[:3,:3], pose_gt[:3,3]
        rot = Rotation.from_matrix(R)
        quad = rot.as_quat()
        quad_wxyz = np.array([quad[3], quad[0], quad[1], quad[2]])
        pose = np.concatenate([t, quad_wxyz], 0)
        trj_gt_np.append(pose_gt)
    trj_gt_np = np.stack(trj_gt_np, 0)
    img = plot_pose(trj_gt_np, trj_est_np)

    return img

def eval_pose_metrics(cameras, save_dir):
    trj_est_np, trj_gt_np = [], []
    
    for camera in cameras:
        pose_est = np.linalg.inv(getWorld2View_np(camera.R, camera.T))
        trj_est_np.append(pose_est)
    trj_est_np = np.stack(trj_est_np, 0)
        
    for camera in cameras:
        pose_gt = np.linalg.inv(getWorld2View_np(camera.R_gt, camera.T_gt))
        trj_gt_np.append(pose_gt)
    trj_gt_np = np.stack(trj_gt_np, 0)
    
    eval_pose(trj_est_np, trj_gt_np, save_dir)

def eval_pose(trj_est_np, trj_gt_np, save_dir):    
    poses_pred = torch.from_numpy(trj_est_np)
    poses_gt_c2w = torch.from_numpy(trj_gt_np)
    poses_gt = poses_gt_c2w[:len(poses_pred)].clone()

    # Align scale first (we do this because scale differs a lot)
    trans_gt_align, trans_est_align, _ = align_pose(poses_gt[:, :3, -1].numpy(),
                                                     poses_pred[:, :3, -1].numpy())
    poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
    poses_pred[:, :3, -1] = torch.from_numpy(trans_est_align)

    # Align full pose (rotation + translation)
    c2ws_est_aligned = align_ate_c2b_use_a2b(poses_pred, poses_gt)

    # Compute ATE and RPE (including per-view errors)
    ate, per_view_ate = compute_ATE(poses_gt.cpu().numpy(),
                                    c2ws_est_aligned.cpu().numpy())
    rpe_trans, rpe_rot, per_view_trans, per_view_rot = compute_rpe(
        poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())

    print("{0:.3f}".format(rpe_trans * 100),
          '&', "{0:.3f}".format(rpe_rot * 180 / np.pi),
          '&', "{0:.3f}".format(ate))

    # Write summary to txt
    with open(f"{save_dir}/pose_eval.txt", 'w') as f:
        f.write("RPE_trans: {:.03f}, RPE_rot: {:.03f}, ATE: {:.03f}\n".format(
            rpe_trans * 100,
            rpe_rot * 180 / np.pi,
            ate))

    # Write per-view errors to CSV
    csv_path = os.path.join(save_dir, 'pose_per_view_ate.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['FrameIndex', 'ATE'])
        for idx in range(len(per_view_ate)):
            writer.writerow([idx, per_view_ate[idx]])
    
    csv_path = os.path.join(save_dir, 'pose_per_view_rpe.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['FrameIndex', 'RPE_Trans', 'RPE_Rot'])
        for idx in range(len(per_view_trans)):
            writer.writerow([idx, per_view_trans[idx], per_view_rot[idx]])
        
def align_pose(pose1, pose2):
    mtx1 = np.array(pose1, dtype=np.double, copy=True)
    mtx2 = np.array(pose2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = mtx2 * s

    return mtx1, mtx2, R

def plot_pose(ref_poses, est_poses):
    plt.rc('legend', fontsize=20)  # using a named size
    ref_poses = [pose for pose in ref_poses]
    if isinstance(est_poses, dict):
        est_poses = [pose for k, pose in est_poses.items()]
    else:
        est_poses = [pose for pose in est_poses]
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True,
                           correct_only_scale=False)

    fig = plt.figure()
    traj_by_label = {
        "Ours": traj_est_aligned,
        "Ground-truth": traj_ref
    }
    plot_mode = plot.PlotMode.xyz
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.zaxis.set_tick_params(labelleft=False)
    colors = ['r', 'b']
    styles = ['-', '--']

    for idx, (label, traj) in enumerate(traj_by_label.items()):
        plot.traj(ax, plot_mode, traj,
                  styles[idx], colors[idx], label)

    ax.view_init(elev=10., azim=45)
    plt.tight_layout()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,1:]
    plt.close(fig)

    return img

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)

def vis_depth(depth):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    percentile = 99
    eps = 1e-10

    lo_auto, hi_auto = weighted_percentile(
        depth, np.ones_like(depth), [50 - percentile / 2, 50 + percentile / 2])
    lo = None or (lo_auto - eps)
    hi = None or (hi_auto + eps)
    curve_fn = lambda x: 1/x + eps

    depth, lo, hi = [curve_fn(x) for x in [depth, lo, hi]]
    depth = np.nan_to_num(
            np.clip((depth - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))
    
    depth = (depth * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth