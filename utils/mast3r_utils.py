# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os
sys.path.append('./submodules/mast3r')

import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.image_pairs import make_pairs
from dust3r.utils.device import to_numpy
import torchvision.transforms as tfm
import py3_wget
import cv2
import torch.nn.functional as F

class Mast3rMatcher:
    model_path = "./submodules/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    vit_patch_size = 16

    def __init__(self, device="cuda"):
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.device = device
        self.download_weights()
        self.model = AsymmetricMASt3R.from_pretrained(self.model_path).to(device)

    @staticmethod
    def download_weights():
        url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if not os.path.isfile(Mast3rMatcher.model_path):
            print("Downloading Master(ViT large)... (takes a while)")
            py3_wget.download_file(url, Mast3rMatcher.model_path)

    def preprocess(self, img):
        _, h, w = img.shape
        orig_shape = h, w
        img = self.resize_to_divisible(img, self.vit_patch_size)
        img = self.normalize(img).unsqueeze(0)
        return img, orig_shape

    def resize_to_divisible(self, img: torch.Tensor, divisible_by: int = 14) -> torch.Tensor:
        h, w = img.shape[-2:]
        divisible_h = round(h / divisible_by) * divisible_by
        divisible_w = round(w / divisible_by) * divisible_by
        return tfm.functional.resize(img, [divisible_h, divisible_w], antialias=True)

    def global_align(self, paths, intrinsic=None):
        scenegraph_type = 'complete'
        winsize = 1
        win_cyclic = False
        refid = 0
        lr1 = 0.07
        niter1 = 1000
        lr2 = 0.014
        niter2 = 400
        min_conf_thr = 1.5
        shared_intrinsics = True
        matching_conf_thr = 5.0
        optim_level = 'refine+depth'
        scene_scale = 10.0
        
        scene_graph_params = [scenegraph_type]
        if scenegraph_type in ["swin", "logwin"]:
            scene_graph_params.append(str(winsize))
        elif scenegraph_type == "oneref":
            scene_graph_params.append(str(refid))
        if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
            scene_graph_params.append('noncyclic')
        scene_graph = '-'.join(scene_graph_params)

        cache_dir = "tmp/cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        images = load_images(paths, size=512)
        pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)

        scene = sparse_global_alignment(
            paths, pairs, cache_dir,
            self.model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, 
            device=self.device,
            opt_depth='depth' in optim_level, 
            shared_intrinsics=shared_intrinsics,
            matching_conf_thr=matching_conf_thr
        )
        
        pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=False)
        pts3d = to_numpy(pts3d)
        confs = to_numpy(confs)
        imgs = np.array(scene.imgs)
        W, H = scene.imgs[0].shape[1], scene.imgs[0].shape[0]
        world2cam = torch.linalg.inv(scene.get_im_poses().detach())
        depthmaps = to_numpy(depthmaps)
        focals = scene.get_focals().detach()
        avg_focal = focals.mean().item()

        if intrinsic is not None:
            intrinsics = np.array([intrinsic for _ in range(len(pts3d))])
        else:
            intrinsic = np.zeros((3, 3))
            intrinsic[0, 0] = avg_focal
            intrinsic[1, 1] = avg_focal
            intrinsic[2, 2] = 1
            intrinsic[0, 2] = W / 2
            intrinsic[1, 2] = H / 2
            intrinsics = np.array([intrinsic for _ in range(len(pts3d))])

        depth_maps = []
        print(f'>> Confidence-aware Ranking...')
        avg_conf_scores = confs.mean(axis=(1, 2))
        sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        sorted_conf_avg_conf_scores = avg_conf_scores[sorted_conf_indices]
        print("Sorted indices:", sorted_conf_indices)
        print("Sorted average confidence scores:", sorted_conf_avg_conf_scores)

        print(f'>> Calculate the co-visibility mask...')
        overlapping_masks = compute_co_vis_masks(
            sorted_conf_indices, depthmaps, pts3d, intrinsics, 
            world2cam.cpu().numpy(), imgs.shape, depth_threshold=0.01
        )
        overlapping_masks = ~overlapping_masks

        if intrinsic is None:
            intrinsic = np.zeros((3, 3))
            intrinsic[0, 0] = avg_focal
            intrinsic[1, 1] = avg_focal
            intrinsic[2, 2] = 1
            intrinsic[0, 2] = W / 2
            intrinsic[1, 2] = H / 2
            scale = 10
            pts3d = pts3d * scale
            world2cam[:, :3, 3] = world2cam[:, :3, 3] * scale
        else:
            scale = 1

        for i, pts in enumerate(pts3d):
            depth_map = depthmaps[i].reshape(H, W) * scale
            depth_maps.append(depth_map)

        overlapping_masks = to_numpy(overlapping_masks).reshape(len(pts3d), -1)
        pts3d = np.concatenate([p[m] for p, m in zip(pts3d, overlapping_masks)])
        pts3d = pts3d.reshape(-1, 3)

        return pts3d, world2cam, depth_maps, avg_focal

    def _forward(self, img0, img1, intrinsic=None):
        img0, img0_orig_shape = self.preprocess(img0)
        img1, img1_orig_shape = self.preprocess(img1)

        img_pair = [
            {"img": img0, "idx": 0, "instance": 0, "true_shape": np.int32([img0.shape[-2:]])},
            {"img": img1, "idx": 1, "instance": 1, "true_shape": np.int32([img1.shape[-2:]])},
        ]
        output = inference([tuple(img_pair)], self.model, self.device, batch_size=1, verbose=False)
        
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]
        
        desc1, desc2 = pred1["desc"].squeeze(0).detach(), pred2["desc"].squeeze(0).detach()
        desc_conf1, desc_conf2 = pred1["desc_conf"].squeeze(0).detach(), pred2["desc_conf"].squeeze(0).detach()
        pts3d1, pts3d2 = pred1["pts3d"].squeeze(0).detach(), pred2["pts3d_in_other_view"].squeeze(0).detach()
        conf1, conf2 = pred1["conf"].squeeze(0).detach(), pred2["conf"].squeeze(0).detach()

        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8, device=self.device, dist="dot", block_size=2**13
        )

        H0, W0 = view1["true_shape"][0]
        H1, W1 = view2["true_shape"][0]
        
        valid_matches_im0 = (
            (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) &
            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
        )
        valid_matches_im1 = (
            (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) &
            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
        )

        valid_matches = valid_matches_im0 & valid_matches_im1
        mkpts0, mkpts1 = matches_im0[valid_matches], matches_im1[valid_matches]
        
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = torch.tensor(mkpts0)
        mkpts1 = torch.tensor(mkpts1)
        mkpts0 = to_normalized_coords(mkpts0, H0, W0) * 2 - 1
        mkpts1 = to_normalized_coords(mkpts1, H1, W1) * 2 - 1

        if intrinsic is not None:
            depth1 = get_depth_map(pts3d1, intrinsic)
            depth1 = torch.from_numpy(depth1).float().cuda().detach()
            depth1 = F.interpolate(depth1.unsqueeze(0).unsqueeze(0), size=(img0_orig_shape[0], img0_orig_shape[1]), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            depth2 = get_depth_map(pts3d2, intrinsic)
            depth2 = torch.from_numpy(depth2).float().cuda().detach()
            depth2 = F.interpolate(depth2.unsqueeze(0).unsqueeze(0), size=(img1_orig_shape[0], img1_orig_shape[1]), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            return mkpts0, mkpts1, desc_conf1, desc_conf2, pts3d1, pts3d2, conf1, conf2, depth1, depth2
        
        return mkpts0, mkpts1, desc_conf1, desc_conf2, pts3d1, pts3d2, conf1, conf2

def to_numpy(x: torch.Tensor | np.ndarray | dict | list) -> np.ndarray:
    if isinstance(x, list):
        return np.array([to_numpy(i) for i in x])
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

def to_normalized_coords(pts: torch.Tensor, height: int, width: int):
    assert pts.shape[-1] == 2, f"input to `to_normalized_coords` should be shape (N, 2), input is shape {pts.shape}"
    pts = pts.to(dtype=torch.float32)
    pts[:, 0] /= width
    pts[:, 1] /= height
    return pts

def get_depth_map(pts3d, intrinsic):
    W, H = pts3d.shape[1], pts3d.shape[0]
    intrinsic[0, 2] = W / 2
    intrinsic[1, 2] = H / 2
    pixels = np.mgrid[:W, :H].T.astype(np.float32).reshape(-1, 2)
    pts3d = pts3d.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
    
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        pts3d, pixels, intrinsic, None, 
        iterationsCount=100, 
        reprojectionError=5, 
        flags=cv2.SOLVEPNP_SQPNP
    )

    R, _ = cv2.Rodrigues(rotation_vector)
    translation_vector = translation_vector.reshape(3)

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = translation_vector

    pts3d = transform_pc_to_cam_coord(pts3d.reshape(-1, 3), pose).reshape(H, W, 3)
    pose = np.linalg.inv(pose)
    depth_map = get_depth_from_3d_points(pts3d.reshape(-1, 3), np.linalg.inv(pose)).reshape(H, W)
    
    return depth_map

def transform_pc_to_cam_coord(points, c2w):
    w2c = np.linalg.inv(c2w)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    points_camera_homogeneous = points_homogeneous.dot(w2c.T)
    return points_camera_homogeneous[:, :3]

def get_depth_from_3d_points(pts_3d, w2c_pose):
    pts_3d_h = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
    pts_camera_h = (w2c_pose @ pts_3d_h.T).T
    return pts_camera_h[:, 2]

def compute_co_vis_masks(sorted_conf_indices, depthmaps, pointmaps, camera_intrinsics, extrinsics_w2c, image_sizes, depth_threshold=0.1):
    num_images, h, w, _ = image_sizes
    pointmaps = pointmaps.reshape(num_images, h, w, 3)
    overlapping_masks = np.zeros((num_images, h, w), dtype=bool)
    
    for i, curr_map_idx in enumerate(sorted_conf_indices):
        if i == 0:
            continue

        idx_before = sorted_conf_indices[:i]
        points_before = pointmaps[idx_before].reshape(-1, 3)
        depths_before = depthmaps[idx_before].reshape(-1)    
        curr_depth_map = depthmaps[curr_map_idx].reshape(h, w)

        depths_before = normalize_depth(depths_before)
        curr_depth_map = normalize_depth(curr_depth_map)

        before_mask = cal_co_vis_mask(points_before, depths_before, curr_depth_map, depth_threshold, camera_intrinsics[curr_map_idx], extrinsics_w2c[curr_map_idx])
        overlapping_masks[curr_map_idx] = before_mask
        
    return overlapping_masks

def normalize_depth(depth_map):
    return (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

def cal_co_vis_mask(points, depths, curr_depth_map, depth_threshold, camera_intrinsics, extrinsics_w2c):
    h, w = curr_depth_map.shape
    overlapping_mask = np.zeros((h, w), dtype=bool)
    points_2d, _ = project_points(points, camera_intrinsics, extrinsics_w2c)
    
    valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
    valid_points_2d = points_2d[valid_points].astype(int)
    valid_depths = depths[valid_points]

    x_coords, y_coords = valid_points_2d[:, 0], valid_points_2d[:, 1]
    depth_differences = np.abs(valid_depths - curr_depth_map[y_coords, x_coords])
    consistent_depth_mask = depth_differences < depth_threshold
    overlapping_mask[y_coords[consistent_depth_mask], x_coords[consistent_depth_mask]] = True

    return overlapping_mask

def project_points(points_3d, intrinsics, extrinsics):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_camera = np.dot(extrinsics, points_3d_homogeneous.T).T
    points_2d_homogeneous = np.dot(intrinsics, points_camera[:, :3].T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]
    depths = points_camera[:, 2]
    return points_2d, depths
