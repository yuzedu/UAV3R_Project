# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchmetrics.functional.regression import pearson_corrcoef

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def depth_loss(depth_gt, depth_pred, weight=None):
    assert depth_gt.shape == depth_pred.shape
    return torch.abs(depth_gt - depth_pred).mean()

def depth_loss_pearson(depth_gt, depth_pred, weight=None):
    assert depth_gt.shape == depth_pred.shape
    depth_gt = depth_gt.reshape(-1)
    depth_pred = depth_pred.reshape(-1)
    if weight is None:
        return (1 - pearson_corrcoef(depth_gt, depth_pred))
    else:
        return (1 - pearson_corrcoef(depth_gt * weight, depth_pred * weight))


def correspondence_2d_loss(kp0, kp1, conf, rendered_depth, view2_world_transform, view1_world_transform, view2_intrinsic):
    """
    Compute 2D correspondence loss between two views.
    
    Args:
        kp0: Keypoints in view1 (N, 2) - normalized coordinates [-1, 1]
        kp1: Keypoints in view2 (N, 2) - normalized coordinates [-1, 1]
        conf: Confidence scores for keypoints (N,)
        rendered_depth: Rendered depth map from view2
        view2_world_transform: World to view2 transform matrix
        view1_world_transform: World to view1 transform matrix
        view2_intrinsic: Intrinsic matrix of view2
    
    Returns:
        loss_2d: 2D correspondence loss
    """
    from utils.graphics_utils import warping
    
    # Convert normalized coordinates to [0, 1] range
    xy0 = kp0 / 2 + 0.5
    xy1 = warping(rendered_depth, view2_world_transform, view1_world_transform.detach(), view2_intrinsic, kp1)
    xy1 = xy1 / 2 + 0.5
    
    # Create mask for valid coordinates
    mask = torch.logical_and(xy1 > 0., xy1 < 1.).all(dim=-1)
    
    # Apply mask to get valid correspondences
    xy0, xy1, conf = xy0[mask], xy1[mask], conf[mask]
    
    # Compute L1 loss weighted by confidence
    if len(xy0) > 0:
        loss_2d = ((xy0.detach() - xy1).abs() * conf[:, None]).mean()
    else:
        loss_2d = torch.tensor(0.0, device=kp0.device, requires_grad=True)
    
    return loss_2d 

