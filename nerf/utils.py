import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips

from basicsr.losses.basic_loss import PerceptualLoss, WeightedTVLoss
from basicsr.losses.gan_loss import GANLoss 

# from .lib import sr_esrnet, sr_unetdisc
# from .lib.masked_adam import MaskedAdam


# landmark loss
import torchvision.transforms as transforms
import torchvision.models as models
import importlib
import sys

from configparser import ConfigParser
import torchvision.transforms.functional as F_v


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

# copied from pytorch3d
def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


@torch.cuda.amp.autocast(enabled=False)
def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str='XYZ') -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    # print(euler_angles, euler_angles.dtype)

    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@torch.cuda.amp.autocast(enabled=False)
def convert_poses(poses):
    # poses: [B, 4, 4]
    # return [B, 3], 4 rot, 3 trans
    out = torch.empty(poses.shape[0], 6, dtype=torch.float32, device=poses.device)
    out[:, :3] = matrix_to_euler_angles(poses[:, :3, :3])
    out[:, 3:] = poses[:, :3, 3]
    return out

@torch.cuda.amp.autocast(enabled=False)
def get_bg_coords(H, W, device):
    X = torch.arange(H, device=device) / (H - 1) * 2 - 1 # in [-1, 1]
    Y = torch.arange(W, device=device) / (W - 1) * 2 - 1 # in [-1, 1]
    xs, ys = custom_meshgrid(X, Y)
    bg_coords = torch.cat([xs.reshape(-1, 1), ys.reshape(-1, 1)], dim=-1).unsqueeze(0) # [1, H*W, 2], in [-1, 1]
    return bg_coords

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

# todo: 下面三个函数明显可以合并起来
@torch.cuda.amp.autocast(enabled=False)
def get_rays_torch_ngp(poses, intrinsics, H, W, N=-1):
    device = poses.device
    poses = poses.view(-1,4,4)
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    zs = torch.ones_like(i)

    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)

    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    rays_d = rays_d.view(H,W,3)
    rays_o = rays_o.view(H,W,3)

    return rays_o, rays_d


@torch.cuda.amp.autocast(enabled=False)
def get_rays_of_a_view(H, W, K, c2w, ndc):
    poses = c2w
    intrinsics = np.array([K[0][0], K[1][1], K[0][2], K[1][2]])
    rays_o, rays_d = get_rays_torch_ngp(poses, intrinsics, H, W, N=-1)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc: # Always False now
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

@torch.cuda.amp.autocast(enabled=False)
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(H=H, W=W, K=K, c2w=c2w, ndc=ndc)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1, importance_map=None, rects=None, train_a2e=False, start_inds=None, torso_patch=False):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        # 训练a2e网络的时候，采样整个脸部的光线
        if train_a2e:
            xmin, xmax, ymin, ymax = rects['face_rect'][0].tolist()
            # print(xmin, ymin, xmax, ymax)
            N = (xmax - xmin) * (ymax - ymin)
            mask = torch.zeros(H, W, dtype=torch.bool, device=device)
            mask[xmin:xmax, ymin:ymax] = 1
            inds = torch.where(mask.view(-1))[0] # [nzn]
            inds = inds.unsqueeze(0) # [1, N]
        # 训练超采网络的时候，需要高分辨率patch，这里intrinsics, H, w, downscale都需要对应高分辨率下的参数
        elif start_inds is not None:
            inds = start_inds # 分辨率整除不了downscale可能引入量化误差
            # print(inds.shape)
            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]
            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten
            inds = inds.expand([B, patch_size * patch_size])
        elif torso_patch:
            num_patch = 1
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, patch_size * patch_size])
            # print(inds.shape)
        else:
            N = min(N, H*W)
            # if use patch-based sampling, ignore error_map
            # patch采样的时候，走如下逻辑
            if patch_size > 1:
                # random sample left-top cores.
                # print(rects['face_rect'][0], type(rects['face_rect'][0]))
                x_min_face, x_max_face, y_min_face, y_max_face = rects['face_rect'][0].tolist()
                x_min_lip, x_max_lip, y_min_lip, y_max_lip = rects['lip_rect'][0].tolist()
                # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
                # 2048条光线，patch_size=16, 一共8个patch，50%采嘴，50%采脸
                num_patch = 1 # (N // 2) // (patch_size ** 2) // 2
                # 嘴的部分
                # print(x_min_face, y_min_face, x_max_face, y_max_face, num_patch)
                # print(x_min_lip, y_min_lip, x_max_lip, y_max_lip)
                inds_x = torch.randint(x_min_lip, max(x_min_lip + 1, x_max_lip - patch_size), size=[num_patch], device=device)
                inds_y = torch.randint(y_min_lip, max(y_min_lip + 1, y_max_lip - patch_size), size=[num_patch], device=device)
                inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

                results['start_ind_lip'] = inds

                # create meshgrid for each patch
                pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
                offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

                inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
                inds = inds.view(-1, 2) # [N, 2]
                inds = inds[:, 0] * W + inds[:, 1] # [N], flatten
                inds_lip = inds.expand([B, patch_size * patch_size])

                # 脸的部分
                inds_x = torch.randint(x_min_face, max(x_min_face + 1, x_max_face - patch_size), size=[num_patch], device=device)
                inds_y = torch.randint(y_min_face, max(y_min_face + 1, y_max_face - patch_size), size=[num_patch], device=device)
                inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

                # create meshgrid for each patch
                pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
                offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

                inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
                inds = inds.view(-1, 2) # [N, 2]
                inds = inds[:, 0] * W + inds[:, 1] # [N], flatten
                inds_face = inds.expand([B, patch_size * patch_size])

                inds = torch.cat((inds_lip, inds_face), dim=-1) #.to(device)


        
            # 不开error map
            elif error_map is None:
                # 不开error map和importance map的时候
                if importance_map is None:
                    inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
                    inds = inds.expand([B, N])
                # 不开error map, 开importance map
                else:

                    importance_map = importance_map.squeeze(2)
                    # print(importance_map.shape,)
                    inds_coarse = torch.multinomial(importance_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)
                    # map to the original resolution with random perturb.
                    inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
                    sx, sy = H / 128, W / 128
                    inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
                    inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
                    inds = inds_x * W + inds_y

            # 开error map
            else:
                importance_map = importance_map.squeeze(2)
                inds_coarse = torch.multinomial(importance_map.to(device) + error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)
                # map to the original resolution with random perturb.
                inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
                sx, sy = H / 128, W / 128
                inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
                inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
                inds = inds_x * W + inds_y
                results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])
    
    # print(inds)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['inds'] = inds

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def rel_dist_lip(landmarks):
    lip_right = landmarks[[57, 51, 48, 60, 61, 62, 63], :]
    lip_left = landmarks[[8, 33, 54, 64, 67, 66, 65], :]
    dis = torch.sqrt(((lip_right - lip_left) ** 2).sum(1))
    return dis


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'



class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=10, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        # perceptual loss
        if self.opt.patch_size > 1:
            # self.sr_forward_flag = False
            # import lpips
            # self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)
            pcplayer_weight = {
                'conv1_2': 0,
                'conv2_2': 0,
                'conv3_4': 1,
                'conv4_4': 1,
                'conv5_4': 1
            }
            self.criterion_lpips = PerceptualLoss(layer_weights=pcplayer_weight, vgg_type='vgg19', perceptual_weight=0.5, style_weight=0.2).to(self.device)
            self.criterion_tv = WeightedTVLoss(loss_weight=0.01)

        if self.opt.use_sr:
            # 网络结构定义
            dim_rend = 3
            sr_ratio = opt.downscale
            num_cond = 1
            self.net_sr = sr_esrnet.SFTNet(n_in_colors=dim_rend, scale=sr_ratio, num_feat=64, num_block=2, num_grow_ch=32, num_cond=num_cond, dswise=False).to(self.device)
            # ftsr_path = '/home/zhangyan/dyngp/ckpts/RealESRNet_x4plus.pth'
            self.net_sr.load_network(load_path=opt.sr_path, device=device, strict=False)

            # 优化器
            param_sr = []
            lrate_srnet = 2e-4
            param_sr.append({'params': self.net_sr.parameters(), 'lr': lrate_srnet, 'kname': 'srnet', 'skip_zero_grad': (False)})
            self.optimizer_sr = MaskedAdam(param_sr)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        
        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file


    def train_step(self, data):
        """
        目前设定是只在use_sr，且<=pretrained_epoch
        或者not use_sr的时候使用
        """
        # B是B张图片，N=H*W
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        rays_exprs = data['rays_exprs']# [B, N, 79]
        bg_coords = data['bg_coords'] # [1, N, 2]
        poses = data['poses'] # [B, 6]
        face_mask = data['face_mask'] # [B, N]


        # print(rays_o.shape, rays_d.shape, rays_exprs.shape)

        if self.opt.use_patch_loss:
            rays_patch_o = data['rays_patch']['rays_o']
            rays_patch_d = data['rays_patch']['rays_d']

        # images = data['images'] # [B, N, 3/4]
        if not self.opt.torso:
            images = data['images'] # [B, N, 3]
        else:
            images = data['bg_torso_color']

        B, N, C = images.shape
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        
        args_dict = vars(self.opt)
        if self.opt.use_bc:
            bg_color = data['bc']
        # use latent code
        if self.opt.use_latent_code:
            args_dict['index'] = data['index']
            args_dict['mode'] = 'train'

        outputs = self.model.render(rays_o, rays_d, rays_exprs, bg_coords, poses, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, **args_dict)

        if self.opt.use_patch_loss:
            if self.opt.use_bc:
                bg_color = data['bc_patch']

            bg_coords = data['bg_coords_patch'] # [1, N, 2]
            outputs_patch = self.model.render(rays_patch_o, rays_patch_d, rays_exprs, bg_coords, poses, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, **args_dict)

        # pred_rgb = outputs['image']

        if not self.opt.torso:
            pred_rgb = outputs['image']
        else:
            pred_rgb = outputs['torso_color']

        loss_latent_code_lambda = self.opt.loss_latent_lambda
        loss_weights_lambda = self.opt.loss_mask_lambda
        loss_patch_lambda = self.opt.loss_patch_lambda

        # latent_code_loss
        if self.opt.use_latent_code:
            loss_latent_code = torch.norm(outputs['latent_code']) * loss_latent_code_lambda
        else:
            loss_latent_code = None

        # weight loss
        # print(self.opt.use_mask_loss)
        if self.opt.use_mask_loss:
            alpha = outputs['weights_sum'] if not self.opt.torso else outputs['torso_alpha']
            # print(data['masks_resize'].shape, data['masks'].shape)
            loss_weights = background_loss = F.l1_loss(
                outputs['weights_sum'], data['masks'].squeeze() / 255., reduction='none'
                ).unsqueeze(0) * loss_weights_lambda
        else:
            loss_weights = None
        # print(loss_weights.shape) 4096
        

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)
        # print(loss.shape, loss_latent_code.shape, loss_weights.shape) torch.Size([1, 4096]) torch.Size([]) torch.Size([4096])
        loss = loss + loss_latent_code if self.opt.use_latent_code else loss# [B, N, 3] --> [B, N]
        loss = loss + loss_weights if self.opt.use_mask_loss else loss# [B, N, 3] --> [B, N]

        # patch-based rendering
        if self.opt.use_patch_loss:
            if not self.opt.torso:
                pred_rgb_patch = outputs_patch['image']
            else:
                pred_rgb_patch = outputs_patch['torso_color']
            gt_rgb_patch = data['images_patch']

            # pred_rgb_patch = outputs_patch['image']
            
            gt_rgb_patch = gt_rgb_patch.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb_patch = pred_rgb_patch.view(-1, self.opt.patch_size, self.opt.patch_size, 3).permute(0, 3, 1, 2).contiguous()

            """
            # patch出来看起来还是有点问题
            if not os.path.exists("/mnt/home/my-blendshape-nerf/exp/hys3s_expr/test_nbs_origin_512_fast_update/debug"):
                os.mkdir("/mnt/home/my-blendshape-nerf/exp/hys3s_expr/test_nbs_origin_512_fast_update/debug")
            # DEBUG:
            print(gt_rgb_patch.shape)
            print(pred_rgb_patch.shape)

            im_pred = (pred_rgb_patch.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)[0]
            im_gt = (gt_rgb_patch.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)[0]


            im_pred = cv2.cvtColor(im_pred, cv2.COLOR_BGR2RGB)
            im_gt = cv2.cvtColor(im_gt, cv2.COLOR_BGR2RGB)

            cv2.imwrite(f"/mnt/home/my-blendshape-nerf/exp/hys3s_expr/test_nbs_origin_512_fast_update/debug/patch{str(data['index'][0])}.jpg", im_pred)
            cv2.imwrite(f"/mnt/home/my-blendshape-nerf/exp/hys3s_expr/test_nbs_origin_512_fast_update/debug/gt_patch{str(data['index'][0])}.jpg", im_gt)
            """

            # LPIPS loss [not useful...]
            loss = loss + loss_patch_lambda * self.criterion_lpips(pred_rgb_patch, gt_rgb_patch)[0]
            loss = loss + self.criterion_tv(pred_rgb_patch, gt_rgb_patch)
        
        if not self.opt.torso:
            ambient = outputs['ambient'] # [N], abs sum
            loss_amb = (ambient * (~face_mask.view(-1))).mean()

            # gradually increase it
            lambda_amb = min(self.global_step / self.opt.iters, 1.0) * self.opt.loss_amb_lambda
            # lambda_amb = self.opt.lambda_amb
            loss = loss + lambda_amb * loss_amb

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()

        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss = loss + loss_ws.mean()

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        rays_exprs = data['rays_exprs']# [B, N, 79]
        bg_coords = data['bg_coords'] # [1, N, 2]
        poses = data['poses'] # [B, 6]

        images = data['images']

        B, H, W, C = images.shape
        gt_rgb = images

        # eval with fixed background color
        bg_color = 1
        # print('rays_exprs:',rays_exprs,rays_exprs.shape)
        args_dict = vars(self.opt)
        if self.opt.use_bc:

            bg_color = data['bc']
        if self.opt.use_latent_code:
            args_dict['index'] = data['index']
            args_dict['mode'] = 'val'

        # print(rays_o.shape, bg_color.shape)
        # print(rays_o.shape, data['bc'].shape)

        outputs = self.model.render(rays_o, rays_d, rays_exprs, bg_coords, poses, staged=True, bg_color=bg_color, perturb=False, **args_dict)
        # print(outputs['image'].shape)
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()
        return pred_rgb, pred_depth, gt_rgb, loss


    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        rays_exprs = data['rays_exprs']# [B, N, 79]
        bg_coords = data['bg_coords'] # [1, N, 2]
        poses = data['poses'] # [B, 6]
        # rays_patch = data['rays_patch'] if self.opt.use_patch_loss else None
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        # use latent code
        args_dict = vars(self.opt)
        if self.opt.use_bc:
            bg_color = data['bc']

        if self.opt.use_latent_code:
            args_dict['index'] = data['index']
            args_dict['mode'] = 'test'

        outputs = self.model.render(rays_o, rays_d, rays_exprs, bg_coords, poses, staged=True, bg_color=bg_color, perturb=perturb, **args_dict)

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth


    def load_mouth_data_for_sr(self, data):
        data_mouth = {}
        data_mouth['rays_o'] = data['mouth_train_data']['rays']['rays_o'] # [B, N, 3]
        data_mouth['rays_d'] = data['mouth_train_data']['rays']['rays_d'] # [B, N, 3]
        data_mouth['rays_exprs'] = data['rays_exprs'] # [B, N, 79]
        data_mouth['bg_coords'] = data['mouth_train_data']['bg_coords_patch'] # [1, N, 2]
        data_mouth['poses'] = data['poses'] # [B, 6]
        # data_mouth['target_patch'] = data['mouth_train_data']['images_patch']
        data_mouth['target_4x_patch'] = data['mouth_train_data']['images_patch_4x']
        data_mouth['bc'] = data['mouth_train_data']['bc_patch']

        # for k, v in data_mouth.items():
        #     print(k, v.shape)

        data_mouth['index'] = data['index']
        data_mouth['pr'] = data['pr']
        data_mouth['pc'] = data['pc']

        # print(data_mouth['index'], data_mouth['pr'], data_mouth['pc'])

        return data_mouth

    # def save_mesh(self, save_path=None, resolution=256, threshold=10):

    #     if save_path is None:
    #         save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

    #     self.log(f"==> Saving mesh to {save_path}")

    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)

    #     def query_func(pts):
    #         with torch.no_grad():
    #             with torch.cuda.amp.autocast(enabled=self.fp16):
    #                 sigma = self.model.density(pts.to(self.device))['sigma']
    #         return sigma

    #     vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

    #     mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
    #     mesh.export(save_path)

    #     self.log(f"==> Finished saving mesh.")


    def loss_adjustment(self, epoch):
        """
        论文中的loss调整策略
        - stage1: color + mask + PIPS = 1:1:0，前两个epoch
        - stage2: color + mask + PIPS = 1:0:0，（2-7 or 3-7）epoch
        - stage3: 随机采样光线 + 基于patch采样光线，对于随机光线color + mask + PIPS = 1:0:0，
                  对于patch光线（嘴部区域0.5概率采样，其他区域0.5概率采样），color + mask + PIPS = 0.1:0:0.1，没说epoch
        - 自己增加了一个latent code loss, 权重设小一点，全程加上
        """
        step1 = 3
        step2 = 7
        step3 = self.opt.nerf_pretrained_epoch

        self.opt.loss_latent_code_lambda = 0.001

        if epoch <= step1:
            # mask loss
            self.opt.loss_photo_lambda = 1.0
            self.opt.loss_mask_lambda = 1.0 * 0.2
            self.opt.loss_patch_lambda = 0.0
            # self.sr_forward_flag  = False
        elif epoch > step1 and epoch <= step2:
            # 只用photo loss
            self.opt.loss_photo_lambda = 1.0
            self.opt.loss_mask_lambda = 0.0
            self.opt.loss_patch_lambda = 0.0
            # self.sr_forward_flag = False
        elif epoch > step2 and epoch <= step3:
            # perceptual loss
            self.opt.loss_photo_lambda = 1.0
            self.opt.loss_mask_lambda = 0.0
            self.opt.loss_patch_lambda = 0.001 * 0.5 # 可能还是有点大？
            # self.sr_forward_flag = False
        else:
            if self.opt.use_sr:
                self.opt.loss_photo_lambda = 0.0
                self.opt.loss_mask_lambda = 0.0
                self.opt.loss_patch_lambda = 0.0 # 0.1太大了 0.001收敛有点慢
                self.opt.loss_sr_patch_lambda = 0.005
                self.opt.loss_sr_photo_lambda = 1.0
                # self.sr_forward_flag = True


    def train(self, train_loader, train_loader_sr, valid_loader, max_epochs):

        # self.model.load_min_max_expr(self.opt.path, self.device)

        self.log(self.opt)
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            # TODO: 这里如果train a2e > 15 epoch会出现逻辑问题，看看怎么实现好
            if self.opt.use_sr and self.epoch == self.opt.nerf_pretrained_epoch + 1:
                self.log("##--------------setting nerf parameters to no grad--------------##")
                for param in self.model.named_parameters():
                    param[1].requires_grad = False

            # if self.opt.use_mask_loss and epoch > self.opt.mloss_duration_epoch:
            #     self.opt.use_mask_loss = False
            # 根据epoch数调整loss及权重
            self.loss_adjustment(epoch)

            # TODO: 这里如果train a2e > 15 epoch会出现逻辑问题，看看怎么实现好
            if self.opt.use_sr and train_loader_sr is not None:
                self.train_one_epoch(train_loader_sr)

            else:
                self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        # self.model.load_min_max_expr(self.opt.path, self.device)
        
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True, test_fps=30):

        # self.model.load_min_max_expr(self.opt.path, self.device)

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            suffix = ''

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb{suffix}.mp4'), all_preds, fps=test_fps, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth{suffix}.mp4'), all_preds_depth, fps=test_fps, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()
        
        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    args_dict = {'use_latent_code': self.opt.use_latent_code, 'mode': 'train'}
                    if self.opt.use_latent_code:
                        args_dict['index'] = data['index']
                    self.model.update_extra_state(data, **args_dict)

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            if torch.any(torch.isnan(loss)):
                print("Loss is nan! Ignore current step!")
                continue
            
            loss.requires_grad_(True)
            self.scaler.scale(loss).backward()
            # if self.epoch <= self.opt.nerf_pretrained_epoch:

            self.scaler.step(self.optimizer)
            # TODO: 理论上a2e只需要step下面这一步就行，上面不需要
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    # def train_one_epoch_a2e(self, loader):
    #     self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

    #     total_loss = 0
    #     if self.local_rank == 0 and self.report_metric_at_train:
    #         for metric in self.metrics:
    #             metric.clear()
        

    #     # eval()是把model.training设为true, 对参数是否训练无关
    #     # 不过这两项都不需要训练，所以都设为eval就行了
    #     self.model.eval()
    #     self.a2enet.train()
    #     if self.opt.use_sr:
    #         self.net_sr.eval()

    #     # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
    #     # ref: https://pytorch.org/docs/stable/data.html
    #     if self.world_size > 1:
    #         loader.sampler.set_epoch(self.epoch)
        
    #     if self.local_rank == 0:
    #         pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    #     self.local_step = 0

    #     batch_size = 4
    #     loss_batch = 0.

    #     for data in loader:

    #         self.local_step += 1
    #         self.global_step += 1

    #         if self.local_step % batch_size == 1:
    #             self.optimizer_a2e.zero_grad()
    #             loss_batch = 0.

    #         with torch.cuda.amp.autocast(enabled=self.fp16):
    #             # TODO: 梳理逻辑
    #             preds, truths, loss = self.train_step_a2e(data)

    #         loss_batch = loss_batch + loss 
    #         if torch.any(torch.isnan(loss)):
    #             print("Loss is nan! Ignore current step!")
    #             continue

    #         if self.local_step % batch_size == 0:
    #             loss_batch.requires_grad_(True)
    #             self.scaler.scale(loss).backward()
    #             self.optimizer_a2e.step()
    #             self.scaler.update()

    #         if self.scheduler_update_every_step:
    #             self.lr_scheduler.step()

    #         loss_val = loss.item()
    #         total_loss += loss_val

    #         if self.local_rank == 0:
    #             if self.report_metric_at_train:
    #                 for metric in self.metrics:
    #                     metric.update(preds, truths)

    #             if self.use_tensorboardX:
    #                 self.writer.add_scalar("train/loss", loss_val, self.global_step)
    #                 self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

    #             if self.scheduler_update_every_step:
    #                 pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
    #             else:
    #                 pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
    #             pbar.update(loader.batch_size)

    #     if self.ema is not None:
    #         self.ema.update()

    #     average_loss = total_loss / self.local_step
    #     self.stats["loss"].append(average_loss)

    #     if self.local_rank == 0:
    #         pbar.close()
    #         if self.report_metric_at_train:
    #             for metric in self.metrics:
    #                 self.log(metric.report(), style="red")
    #                 if self.use_tensorboardX:
    #                     metric.write(self.writer, self.epoch, prefix="train")
    #                 metric.clear()

    #     if not self.scheduler_update_every_step:
    #         if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #             self.lr_scheduler.step(average_loss)
    #         else:
    #             self.lr_scheduler.step()

    #     self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()
        if self.opt.use_sr:
            self.net_sr.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):

                    if self.opt.use_sr:
                        preds, preds_depth, truths, loss, preds_low = self.eval_step_sr(data)
                    else:
                        preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    if self.opt.use_sr:
                        preds_low = preds_low[0].detach().cpu().numpy()
                        preds_low = (preds_low * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb_low.png'), cv2.cvtColor(preds_low, cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed! (but not perturb the first sample)
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        
        if self.opt.use_latent_code:
            self.model.save_latent_codes(os.path.join(self.opt.workspace, 'checkpoints'))

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density
            state['mean_density_torso'] = self.model.mean_density_torso

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()


            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            if self.opt.use_sr:
                self.net_sr.save_network(self.ckpt_path, f'sresrnet_epoch', self.epoch)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    # if 'density_grid' in state['model']:
                    #     del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    if self.opt.use_sr:
                        self.net_sr.save_network(self.ckpt_path, f'sresrnet_epoch', self.epoch)
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)

        # 如果有训练权重，加载最近一个epoch的权重f
        if self.opt.use_sr:
            model_name = sorted([i for i in os.listdir(os.path.join(self.opt.workspace, 'checkpoints')) 
                                if i.startswith('sresrnet')], key=lambda x: int(x.split('.')[0].split('_')[-1]))
            if len(model_name) == 0:
                print("no trained esrnet, continue...")
            else:
                model_name = model_name[-1]
                self.net_sr.load_network(os.path.join(self.opt.workspace, 'checkpoints', model_name), device=self.device, strict=False)
                print(f"load esrnet pth: {model_name}.")

        # 加载latent codes
        latent_code_path = os.path.join(self.ckpt_path, "latent_codes.ckpt")
        if os.path.exists(latent_code_path) and self.opt.use_latent_code:
            data = torch.load(latent_code_path)
            self.model.latent_codes = data['latent_codes']
            print("latent codes loaded!")

        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
            if 'mean_density_torso' in checkpoint_dict:
                self.model.mean_density_torso = checkpoint_dict['mean_density_torso']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
