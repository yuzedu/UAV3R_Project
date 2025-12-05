# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import torch
from utils.camera_utils import camera_to_JSON
import json
from scipy.interpolate import UnivariateSpline

def save_transforms(cameras, path):
    json_cams = []
    viewpoint_stack = cameras
    for id, view in enumerate(viewpoint_stack):
        json_cams.append(camera_to_JSON(None, view))

    with open(path, 'w') as file:
        json.dump(json_cams, file, indent=2)

def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm

def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )

def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V

def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R.t()
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3].t()
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def filter1d(vec, time, W):
    stepsize = 2 * W + 1
    filtered = np.median(strided_app(vec, stepsize, stepsize), axis=-1)
    pre_smoothed = np.interp(time, time[W:-W:stepsize], filtered)
    return pre_smoothed

def smooth_vec(vec, time, s, median_prefilter):
    if median_prefilter:
        vec = np.stack([
            filter1d(vec[..., 0], time, 5),
            filter1d(vec[..., 1], time, 5),
            filter1d(vec[..., 2], time, 5)
        ], axis=-1)
    smoothed = np.zeros_like(vec)
    for i in range(vec.shape[1]):
        spl = UnivariateSpline(time, vec[..., i])
        spl.set_smoothing_factor(s)
        smoothed[..., i] = spl(time)
    return smoothed

def smooth_poses_spline(poses, st=0.5, sr=4, median_prefilter=True):
    poses = np.asarray(poses)
    assert poses.shape[1:] == (4, 4), "Input must be (N, 4, 4) pose matrices."
    if len(poses) < 30:
        median_prefilter = False
    # Extract 3x4 for smoothing
    poses_3x4 = poses[:, :3, :4].copy()
    # For compatibility with old code, flip x axis before smoothing
    poses_3x4[:, 0] = -poses_3x4[:, 0]
    posesnp = poses_3x4
    scale = 2e-2 / np.median(np.linalg.norm(posesnp[1:, :3, 3] - posesnp[:-1, :3, 3], axis=-1))
    posesnp[:, :3, 3] *= scale
    time = np.linspace(0, 1, len(posesnp)) 
    t = smooth_vec(posesnp[..., 3], time, st, median_prefilter)
    z = smooth_vec(posesnp[..., 2], time, sr, median_prefilter)
    z /= np.linalg.norm(z, axis=-1)[:, None]
    y_ = smooth_vec(posesnp[..., 1], time, sr, median_prefilter)
    x = np.cross(z, y_)
    x /= np.linalg.norm(x, axis=-1)[:, None]
    y = np.cross(x, z)
    smooth_posesnp = np.stack([x, y, z, t], -1)
    poses_3x4[:, 0] = -poses_3x4[:, 0]
    smooth_posesnp[:, 0] = -smooth_posesnp[:, 0]
    smooth_posesnp[:, :3, 3] /= scale
    N = poses.shape[0]
    out = np.zeros((N, 4, 4), dtype=poses.dtype)
    out[:, :3, :4] = smooth_posesnp
    out[:, 3, 3] = 1.0
    return out