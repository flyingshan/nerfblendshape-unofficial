import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import pickle as pkl
import trimesh
import math

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .utils import get_rays, get_training_rays, get_bg_coords, convert_poses, euler_angles_to_matrix

def smooth_camera_path(poses, kernel_size=5):
    # smooth the camera trajectory...
    # poses: [N, 4, 4], numpy array

    N = poses.shape[0]
    K = kernel_size // 2
    
    trans = poses[:, :3, 3].copy() # [N, 3]
    rots = poses[:, :3, :3].copy() # [N, 3, 3]

    for i in range(N):
        start = max(0, i - K)
        end = min(N, i + K + 1)
        poses[i, :3, 3] = trans[start:end].mean(0)
        poses[i, :3, :3] = Rotation.from_matrix(rots[start:end]).mean().as_matrix()

    return poses

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


def get_coord(rect, extra_pixel, H, W):
    """修正脸部，嘴部coordinate"""
    xmin = max(0, rect[1] - extra_pixel)
    xmax = min(H, rect[1] + rect[3] + extra_pixel)
    ymin = max(0, rect[0] - extra_pixel)
    ymax = min(W, rect[0] + rect[2] + extra_pixel)
    return xmin, xmax, ymin, ymax


def rot2euler(R):
    """# 将transform_matrix[:3, :3]转为expr系数的后三位"""
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([y, x, z]) # [NOTE]: 这样和bs_solver提取的欧拉角的顺序是一致的


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=30):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.training = self.type in ['train', 'all', 'trainval']
        self.val = self.type in ['val']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # auto-detect transforms.json and split mode.
        # load nerf-compatible format data.
        with open(os.path.join(self.root_path, 'transforms_all.json'), 'r') as f:
            transform = json.load(f)

        # H和W是训练nerf的gt图像的实际采用的宽高
        # 通过后面读取图片, 并根据自定义的降采样系数来实际得到gt图像的H和W
        self.H = self.W = None 
        image_suffix = 'jpg'
        if self.H is None or self.W is None:
            f_path = os.path.join(self.root_path, f"head_imgs/",str(transform["frames"][0]['img_id']) + f".{image_suffix}")
            image = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE) 
            self.H = image.shape[0] // self.downscale
            self.W = image.shape[1] // self.downscale

        print(f"image size: [H:{self.H}, W:{self.W}]")

        # read images
        frames = transform["frames"]
        self.test_num_a2e = self.opt.test_num



        if self.opt.airkit:
            if not self.opt.smooth_expr:
                self.expr_matrix = np.load(os.path.join(self.root_path, "expr.npy"))
            else:
                self.expr_matrix = np.load(os.path.join(self.root_path, "expr_smooth.npy"))

        if self.opt.test_expr:
            # 采用另一个数据集的pose
            with open(os.path.join(self.root_path, 'transforms_all_test.json'), 'r') as f:
                transform_test = json.load(f)
                frames_test = transform_test["frames"]
                # 按照测试集的pose的旋转矩阵和训练集的pose的平移向量来决定pose
                # 如果测试集长度大于训练集长度，镜像地复用训练集中pose的平移向量
                pose_new = []
                for i in range(len(frames_test)):
                    size = len(frames)
                    turn = i // size
                    res = i % size
                    if turn % 2 == 0:
                        i = res
                    else:
                        i = size - res - 1
                    # 用source驱动target, frames_test驱动frames
                    trans_target = np.array(frames[0]['trans'])
                    rot_inv_source = np.array(frames_test[i]['transform_matrix'])[:3, :3]
                    trans_inv_target = -rot_inv_source @ trans_target
                    pose_source = np.zeros([4, 4])
                    pose_source[:3, :3] = rot_inv_source
                    pose_source[:3, 3] = trans_inv_target
                    pose_new.append(pose_source)

                # 用新的pose
                for i in range(len(frames_test)):
                    frames_test[i]['transform_matrix'] = pose_new[i]

                frames = frames_test

            self.expr_matrix_test = np.load(os.path.join(self.root_path, "expr_test.npy"))

            print(self.expr_matrix.shape, self.expr_matrix_test.shape)

            size = min(self.expr_matrix_test.shape[0], self.expr_matrix.shape[0])
            self.expr_matrix_test = self.expr_matrix_test[:size, :]
            self.expr_matrix = self.expr_matrix[:size, :]
            # 因为目前还使用了pose相关的表情系数（TODO: 考虑放弃这3个表情系数？没有道理模型本身和角度相关）
            self.expr_matrix_test[:, -3:] = self.expr_matrix[:, -3:]
            self.expr_matrix = self.expr_matrix_test
            self.test_num_a2e = 1000


        # 测试时
        if type == 'test':
            # print("val file:",f)
            # frames = frames[:1000] # test on trainset
            if self.opt.test_audio:
                frames = frames[:]
            elif self.opt.test_expr:
                frames = frames[:self.test_num_a2e]
            else:
                frames = frames[-self.test_num_a2e:] # frames[:self.test_num_a2e] + # [TODO: 只为了做demo这样设置] 用训练集的pose, 更加保险
            self.poses = []
            self.images = None
            self.images_resize = None
            self.exprs = []
            # weight loss计算用
            self.masks = None
            self.masks_resize = None
            self.aud = []

            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):

                # load expr parameters.
                if self.opt.airkit:
                    base_expr = self.load_expr_airkit(f)
                else:
                    base_expr = self.load_expr_3DMM(f)
                self.exprs.append(base_expr)
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]

                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                self.poses.append(pose)

            if self.opt.fix_angle:
                self.poses_l1 = []
                vec = np.array([0,0,1])
                for i in range(len(self.poses)):
                    tmp_pose=np.identity(4,dtype=np.float32)
                    pose_base = np.array(transform["frames"][0]["transform_matrix"])[:3, :3]
                    vec = - np.array(transform["frames"][0]["trans"])
                    r1 = Rotation.from_euler('y', -10, degrees=True)
                    pose_base = pose_base @ r1.as_matrix()
                    tmp_pose[:3,:3] = pose_base
                    trans = tmp_pose[:3,:3]@vec
                    tmp_pose[0:3,3] = trans
                    self.poses_l1.append(nerf_matrix_to_ngp(tmp_pose, scale=self.scale, offset=self.offset) )
                self.poses = self.poses_l1


        # 训练或者验证时
        else:

            # for colmap, manually split a valid set (the first frame).
            if type == 'train':
                frames = frames[:-self.test_num_a2e] # debug 1:
            elif type == 'val':
                frames = frames[-self.test_num_a2e::50] # -1
                for i, f in enumerate(frames):
                    # 验证集上gt保存起来
                    f_path = os.path.join(self.root_path, f"gt_imgs/",str(f['img_id']) + f".{image_suffix}")
                    gt_image_val = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
                    os.makedirs(os.path.join(self.opt.workspace, f"validation/"), exist_ok=True)
                    cv2.imwrite(os.path.join(self.opt.workspace, f"validation/", str(i + 1) + '_' + str(f['img_id']) + f"_gt.{image_suffix}"), gt_image_val)

            self.poses = []
            self.images = []
            self.images_resize = [] # 降采样的gt
            self.exprs = []
            # weight loss计算用
            self.masks = []
            self.masks_resize = [] # 降采样的mask
            self.face_rects = []
            self.lip_rects = []
            self.torso_img = []
            self.aud = []

            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                # f_path = os.path.join(self.root_path, f['file_path'])
                if not self.opt.torso:
                    f_path = os.path.join(self.root_path, f"head_imgs/",str(f['img_id']) + f".{image_suffix}")
                else:
                    f_path = os.path.join(self.root_path, f"gt_imgs/",str(f['img_id']) + f".{image_suffix}")

                # load expr parameters.
                if self.opt.airkit:
                    base_expr = self.load_expr_airkit(f)
                else:
                    base_expr = self.load_expr_3DMM(f)
                self.exprs.append(base_expr)

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                
                # mask提取
                f_path_mask = os.path.join(self.root_path, f"head_masks/",str(f['img_id'])+".png")
                mask = cv2.imread(f_path_mask, cv2.IMREAD_GRAYSCALE)
                mask_resize = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_AREA) 

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                image = image.astype(np.float32) / 255 # [H, W, 3/4]
                # resize后面尺寸是(W, H)顺序，输出图像dim是(H, W, C), 注意顺序
                image_resize = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA) 

                if self.training and self.opt.torso:
                    torso_img_path = os.path.join(self.root_path, 'torso_imgs', str(f['img_id']) + '.png')
                    torso_img = cv2.imread(torso_img_path, cv2.IMREAD_UNCHANGED) # [H, W, 4]
                    torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
                    torso_img = torso_img.astype(np.float32) / 255 # [H, W, 3/4]

                    torso_img_resize = cv2.resize(torso_img, (self.W, self.H), interpolation=cv2.INTER_AREA) 
                    self.torso_img.append(torso_img_resize)



                self.poses.append(pose)
                self.images_resize.append(image_resize)
                self.masks_resize.append(mask_resize)

                if self.preload == 0:
                    self.images.append(f_path)
                    self.masks.append(f_path_mask)
                else:
                    self.images.append(image)
                    self.masks.append(mask)

                # 嘴部坐标 脸部坐标
                # 按照xmin, xmax ,ymin, ymax
                extra_pix_face = 0
                extra_pix_lip = 32 # 嘴部区域稍微扩大一点
                xmin, xmax, ymin, ymax = get_coord(f['face_rect'], extra_pix_face, self.H*self.opt.downscale, self.W*self.opt.downscale)
                self.face_rects.append([xmin, xmax, ymin, ymax])
                xmin, xmax, ymin, ymax = get_coord(f['lip_rect'], extra_pix_lip, self.H*self.opt.downscale, self.W*self.opt.downscale)
                self.lip_rects.append([xmin, xmax, ymin, ymax])

            

        # 训练时，要采用提取出的background
        if not self.opt.white_bg:
            self.bc = cv2.imread(os.path.join(self.root_path, f"bc.{image_suffix}"))
            self.bc = cv2.cvtColor(self.bc, cv2.COLOR_BGR2RGB) / 255.
            self.bc_resize = cv2.resize(self.bc, (self.W, self.H), interpolation=cv2.INTER_AREA)  / 255.
        else:
            self.bc = np.ones((self.H*self.downscale, self.W*self.downscale, 3))
            self.bc = self.bc.astype(np.float32)
            self.bc_resize = cv2.resize(self.bc, (self.W, self.H), interpolation=cv2.INTER_AREA) 

        self.poses = np.stack(self.poses, axis=0)
        if self.opt.smooth_path:
            self.poses = smooth_camera_path(self.poses, self.opt.smooth_path_window)
        self.poses = torch.from_numpy(self.poses) # [N, 4, 4]

        if self.images is not None or self.images_resize is not None:
            # print(len(self.image_size), self.image_size[0].shape)
            self.images_resize = torch.from_numpy(np.stack(self.images_resize, axis=0))
            # print("images resize loaded")
            if self.preload > 0:
                self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
            else:
                self.images = np.array(self.images)

        self.exprs = torch.from_numpy(np.stack(self.exprs, axis=0))
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        

        # mask loss
        if self.masks is not None or self.masks_resize is not None:
            if self.preload > 0:
                self.masks = torch.from_numpy(np.stack(self.masks, axis=0)).unsqueeze(-1)
            else:
                self.masks = np.array(self.masks)
            self.masks_resize = torch.from_numpy(np.stack(self.masks_resize, axis=0)).unsqueeze(-1)


        self.bc = torch.from_numpy(self.bc).unsqueeze(0)

        self.bc_resize = torch.from_numpy(self.bc_resize).unsqueeze(0)
        # print("all loaded")


        # lip采样
        if self.training or self.val:
            self.lip_rects = (np.stack(self.lip_rects, axis=0) / self.opt.downscale).astype(np.int32)
            self.face_rects = (np.stack(self.face_rects, axis=0) / self.opt.downscale).astype(np.int32)

        if self.training and self.opt.torso:
            # if self.preload > 0:
            self.torso_img = torch.from_numpy(np.stack(self.torso_img, axis=0)) # [N, H, W, C]
            # else:
            #     self.torso_img = np.array(self.torso_img)


        self.error_map = None

        # 不占显存的，直接加载到GPU
        if self.fp16 and self.opt.color_space != 'linear':
            dtype = torch.half
        else:
            dtype = torch.float
        self.poses        = self.poses.to(self.device)
        self.exprs        = self.exprs.to(self.device)
        self.bc           = self.bc.to(dtype).to(self.device)
        self.bc_resize    = self.bc_resize.to(dtype).to(self.device)
        # 对于占显存较大的项，决定是否要提前加载到GPU
        if self.opt.preload > 1:
            print("preload data to GPU.")
            if self.images is not None or self.images_resize is not None: # 两个应该要么为None，要么都不为None
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                self.images = self.images.to(dtype).to(self.device)
                self.images_resize = self.images_resize.to(dtype).to(self.device)
                self.masks        = self.masks.to(dtype).to(self.device)
                self.masks_resize = self.masks_resize.to(dtype).to(self.device)
                # self.torso_img = self.torso_img.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)


        # load intrinsics
        if 'focal_len' in transform:
            fl_x = (transform['focal_len'] ) / self.downscale
            fl_y = (transform['focal_len'] ) / self.downscale
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / self.downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / self.downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        # super resolution


        self.bg_coords = get_bg_coords(self.H, self.W, self.device) # [1, H*W, 2] in [-1, 1]
        print("Dataset initialization Done!")

    def mirror_index(self, index):
        size = self.poses.shape[0]
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def load_expr_3DMM(self, f):
        with open(os.path.join(self.root_path,f"ori_imgs/", str(f['img_id'])+ '_nl3dmm.pkl'), 'rb') as f_expr: nl3dmm_para_dict = pkl.load(f_expr)
        base_code = nl3dmm_para_dict["code"].detach().unsqueeze(0)
        if self.opt.add_mean:
            base_expr = base_code[:, 100:100 + self.opt.expr_dim - 1].numpy().tolist()[0]
            base_expr.insert(0, 1.0)
        else:
            base_expr = base_code[:, 100:100 + self.opt.expr_dim].numpy().tolist()[0]
        base_expr = np.array(base_expr, dtype=np.float32)
        return base_expr

    def load_expr_airkit(self, f):
        img_id = f['img_id']
        # [NOTE] -1是因为add_mean加了1
        if self.opt.add_mean:
            select_inds = list(range(self.opt.expr_dim - 1)) # list(range(1, 56)) # list(range(1, 62)) # list(range(1, 53)) + list(range(56, 62))
        else:
            select_inds = list(range(self.opt.expr_dim))

        base_code = (self.expr_matrix[img_id][select_inds]).tolist()
        if self.opt.add_mean:
            base_code.insert(0, 1.0)
        base_code = np.array(base_code, dtype=np.float32)
        # print(base_code)
        return base_code

    def collate(self, index):
        B = len(index) # a list of length 1
        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]

        im_map = None

        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, 1, importance_map=im_map) # 如果不preload，可能有速度问题

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'rays_exprs': (self.exprs[index]).type(torch.float32).to(self.device),
        }

        if self.opt.use_patch_loss and self.training:
            rays_patch = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size, \
                                    rects={'lip_rect':self.lip_rects[index], \
                                        'face_rect': self.face_rects[index]},
                                    torso_patch=self.opt.torso)
            results['rays_patch'] = rays_patch

            bg_coords = torch.gather(self.bg_coords, 1, torch.stack(2 * [rays_patch['inds']], -1)) # [1, N, 2]
            results['bg_coords_patch'] = bg_coords

        if self.images is not None:
            images = self.images[index]
            if self.preload == 0:
                images = cv2.imread(images[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
                images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                images = images.astype(np.float32) / 255 # [H, W, 3]
                images = torch.from_numpy(images).unsqueeze(0)
            images = images.to(self.device)
            images_resize = self.images_resize[index].to(self.device)
            # 使用patch loss，额外保存一个gt
            if self.opt.use_patch_loss and self.training:  
                C = images.shape[-1]
                if self.opt.downscale > 1:
                    results['images_patch'] = torch.gather(images_resize.view(B, -1, C), 1, torch.stack(C * [rays_patch['inds']], -1)) # [B, N, 3/4]
                else:
                    results['images_patch'] = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays_patch['inds']], -1)) # [B, N, 3/4]

            # 训练的时候，保存采样的点，测试的时候，不运行这里即直接保存所有image的点
            if self.training:
                C = images.shape[-1]
                # images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
                images_resize = torch.gather(images_resize.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
            if self.opt.downscale > 1:
                results['images'] = images_resize
            else:
                results['images'] = images
            
            if self.val:
                results['images_ori'] = images

        # background sampling
        if self.opt.use_bc:
            bc = self.bc.to(self.device) # 
            bc_resize = self.bc_resize.to(self.device)
            if self.opt.use_patch_loss and self.training:  
                C = images.shape[-1]
                if self.opt.downscale > 1:
                    results['bc_patch'] = torch.gather(bc_resize.view(B, -1, C), 1, torch.stack(C * [rays_patch['inds']], -1)) # [B, N, 3/4]
                else:
                    results['bc_patch'] = torch.gather(bc.view(B, -1, C), 1, torch.stack(C * [rays_patch['inds']], -1)) # [B, N, 3/4]
                # bg_img = torch.gather(bg_img, 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]
            if self.training:
                C = bc.shape[-1]
                bc = torch.gather(bc.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                bc_resize = torch.gather(bc_resize.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            if self.opt.downscale > 1:
                results['bc'] = bc_resize
            else:
                results['bc'] = bc

        # if self.opt.use_latent_code:
        results['index'] = index

        if self.opt.use_mask_loss and self.training:
            # masks = self.masks[index].to(self.device)
            masks = self.masks[index]
            if self.preload == 0:
                masks = cv2.imread(masks[0], cv2.IMREAD_GRAYSCALE)
                masks = torch.from_numpy(masks).unsqueeze(0).unsqueeze(-1)
            masks = masks.to(self.device)
            masks_resize = self.masks_resize[index].to(self.device)
            if self.training:
                C = masks.shape[-1]
                masks = torch.gather(masks.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                masks_resize = torch.gather(masks_resize.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))
            if self.opt.downscale > 1:
                results['masks'] = masks_resize
            else:
                results['masks'] = masks
            face_mask = (results['masks'] == 255).squeeze(-1)
            results['face_mask'] = face_mask

        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        if self.training:
            bg_coords = torch.gather(self.bg_coords, 1, torch.stack(2 * [rays['inds']], -1)) # [1, N, 2]
        else:
            bg_coords = self.bg_coords # [1, N, 2]

        results['bg_coords'] = bg_coords
        results['poses'] = convert_poses(poses) # [B, 6]

        if self.opt.torso and self.training:
            # bg_torso_img = self.torso_img[index].to(self.device)
            bg_torso_img = self.torso_img[index]
            bg_torso_img = bg_torso_img.to(self.device)
            bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bc_resize.squeeze(0) * (1 - bg_torso_img[..., 3:])
            bg_torso_img = bg_torso_img.view(B, -1, 3) # .to(self.device)
            bg_torso_img_rand = torch.gather(bg_torso_img, 1, torch.stack(3 * [rays['inds']], -1)) # [B, N, 3]
            # [NOTE: bug fix: sample torso background]
            results['bg_torso_color'] = bg_torso_img_rand

            if self.opt.use_patch_loss:
                bg_torso_img_patch = torch.gather(bg_torso_img, 1, torch.stack(3 * [rays_patch['inds']], -1)) # [B, N, 3]
                results['bg_torso_color_patch'] = bg_torso_img_patch


        return results

    def dataloader(self, collate='nerf'):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
