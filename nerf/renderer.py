import math
import trimesh
import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from .utils import custom_meshgrid, convert_poses

# EXPR_DIM = 20

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 opt=None,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()
        self.opt = opt
        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        self.density_thresh_torso = opt.density_thresh_torso
        self.torso = opt.torso

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound / 2, -bound, bound, bound / 2, bound]) # /2 from RAD-NERF
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
            
            if self.opt.add_mean:
                self.expr_max = torch.from_numpy(np.load(os.path.join(self.opt.path, 'expr_max.npy'))).float().cuda().unsqueeze(0) # 1, expr_dim
                self.expr_min = torch.from_numpy(np.load(os.path.join(self.opt.path, 'expr_min.npy'))).float().cuda().unsqueeze(0)
            else:
                self.expr_max = torch.from_numpy(np.load(os.path.join(self.opt.path, 'expr_max.npy'))).float().cuda().unsqueeze(0)[:, 1:]
                self.expr_min = torch.from_numpy(np.load(os.path.join(self.opt.path, 'expr_min.npy'))).float().cuda().unsqueeze(0)[:, 1:]

            print("load expr_max successfully:expr_max is :")
            print(self.expr_max)
            print("load expr_min successfully:expr_min is :")
            print(self.expr_min)

        # 2D torso density grid

        if self.torso:
            density_grid_torso = torch.zeros([self.grid_size ** 2]) # [H * H]
            self.register_buffer('density_grid_torso', density_grid_torso)
        self.mean_density_torso = 0

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run_cuda(self, rays_o, rays_d, rays_expr, bg_coords, poses, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        rays_expr = rays_expr.contiguous().view(-1, self.in_dim_expr)

        bg_coords = bg_coords.contiguous().view(-1, 2)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        if kwargs['use_bc']:
            bg_color = bg_color.squeeze().reshape(-1, 3)

        # mix background color
        if bg_color is None:
            bg_color = 1

        # print(rays_expr.dtype)

        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1
            
            # print(rays_o.shape)
            # print(rays_d.shape)

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)
            exprs = rays_expr # [1, expr_dim]
            #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
            result_dict = self(xyzs, dirs, exprs, **kwargs)
            sigmas, rgbs, latent_code, ambient = result_dict['sigma'], result_dict['color'], result_dict['latent_code_res'], result_dict['ambient']
            sigmas = self.density_scale * sigmas

            # weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)
            weights_sum, ambient_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ambient.abs().sum(-1), deltas, rays)
            results['ambient'] = ambient_sum
            results['latent_coef'] = result_dict['latent_coef']
        else:
           
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)
                exprs = rays_expr # [0].expand(xyzs.size()[0], self.in_dim_expr)

                result_dict = self(xyzs, dirs, exprs.reshape(-1, self.in_dim_expr), **kwargs)
                sigmas, rgbs, latent_code, ambient = result_dict['sigma'], result_dict['color'], result_dict['latent_code_res'], result_dict['ambient']

                sigmas = self.density_scale * sigmas

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        # first mix torso with background
        if self.torso:

            # ind_code_torso = None
            
            # 2D density grid for acceleration...
            density_thresh_torso = min(self.density_thresh_torso, self.mean_density_torso)
            occupancy = F.grid_sample(self.density_grid_torso.view(1, 1, self.grid_size, self.grid_size), bg_coords.view(1, -1, 1, 2), align_corners=True).view(-1)
            mask = occupancy > density_thresh_torso

            # masked query of torso
            torso_alpha = torch.zeros([N, 1], device=device)
            torso_color = torch.zeros([N, 3], device=device)

            if mask.any():
                

                if self.opt.torso_head_aware:
                    if random.random() < 0.5:
                        torso_alpha_mask, torso_color_mask, deform = self.forward_torso(bg_coords[mask], poses, None, image[mask], weights_sum.unsqueeze(-1)[mask], **kwargs)
                    else:
                        torso_alpha_mask, torso_color_mask, deform = self.forward_torso(bg_coords[mask], poses, None, **kwargs)
                else:
                    torso_alpha_mask, torso_color_mask, deform = self.forward_torso(bg_coords[mask], poses, None, **kwargs)

                torso_alpha[mask] = torso_alpha_mask.float()
                torso_color[mask] = torso_color_mask.float()

                results['deform'] = deform
            
            # first mix torso with background
            # 训练torso相当于改变bg_color
            bg_color = torso_color * torso_alpha + bg_color * (1 - torso_alpha)

            results['torso_alpha'] = torso_alpha
            results['torso_color'] = bg_color

        image = image + (1 - weights_sum).unsqueeze(-1) *  bg_color
        image = image.view(*prefix, 3)
        image = image.clamp(0, 1)

        depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)

        results['weights_sum'] = weights_sum


        results['depth'] = depth
        results['image'] = image

        if kwargs['use_latent_code']:
            results['latent_code'] = latent_code
        
        return results

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not self.cuda_ray:
            return
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        fx, fy, cx, cy = intrinsic
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # update count 
                            count[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

        print(f'[mark untrained grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, data, decay=0.95, S=128, **kwargs):
        # call before each epoch to update extra states.
        # 每隔16个epoch调用一次这个函数

        if not self.cuda_ray:
            return 

        # full update.
        if self.torso:
            
            tmp_grid_torso = torch.zeros_like(self.density_grid_torso)

            # random pose, random ind_code
            rand_idx = random.randint(0, self.poses.shape[0] - 1)
            pose = convert_poses(self.poses[[rand_idx]]).to(self.density_bitfield.device)
            kwargs['index'] = [rand_idx]

            ind_code = None
            enc_a = None

            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            half_grid_size = 1 / self.grid_size

            for xs in X:
                for ys in Y:
                    xx, yy = custom_meshgrid(xs, ys)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1) # [N, 2], in [0, 128)
                    indices = (coords[:, 1] * self.grid_size + coords[:, 0]).long() # NOTE: xy transposed!
                    xys = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 2] in [-1, 1]
                    xys = xys * (1 - half_grid_size)
                    # add noise in [-hgs, hgs]
                    xys += (torch.rand_like(xys) * 2 - 1) * half_grid_size
                    # query density
                    alphas, _, _ = self.forward_torso(xys, pose, enc_a, **kwargs) # [N, 1]
                    
                    # assign 
                    tmp_grid_torso[indices] = alphas.squeeze(1).float()

            # dilate
            tmp_grid_torso = tmp_grid_torso.view(1, 1, self.grid_size, self.grid_size)
            # tmp_grid_torso = F.max_pool2d(tmp_grid_torso, kernel_size=3, stride=1, padding=1)
            tmp_grid_torso = F.max_pool2d(tmp_grid_torso, kernel_size=5, stride=1, padding=2)
            tmp_grid_torso = tmp_grid_torso.view(-1)

            self.density_grid_torso = torch.maximum(self.density_grid_torso * decay, tmp_grid_torso)
            self.mean_density_torso = torch.mean(self.density_grid_torso).item()

        else:

            tmp_grid_max_expr = torch.zeros_like(self.density_grid)

            for i in range(1, self.opt.expr_dim):
                # 为了速度，随机选其中一些表情维度，取最大值更新

                rays_expr = torch.zeros_like(self.expr_max)
                rays_expr[:, 0] = self.expr_max[:, 0]
                rays_expr[:, i] = self.expr_max[:, i]
                rays_expr = rays_expr.to(self.density_bitfield.device)
                tmp_grid = torch.zeros_like(self.density_grid)

                # full update
                X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
                Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
                Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

                for xs in X:
                    for ys in Y:
                        for zs in Z:
                            
                            # construct points
                            xx, yy, zz = custom_meshgrid(xs, ys, zs)
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                            indices = raymarching.morton3D(coords).long() # [N]
                            xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                            # cascading
                            for cas in range(self.cascade):
                                bound = min(2 ** cas, self.bound)
                                half_grid_size = bound / self.grid_size
                                # scale to current cascade's resolution
                                cas_xyzs = xyzs * (bound - half_grid_size)
                                # add noise in [-hgs, hgs]
                                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                                # query density
                                density_result_dict = self.density(cas_xyzs, rays_expr, **kwargs)
                                sigmas = density_result_dict['sigma'].reshape(-1).detach().to(tmp_grid.dtype)
                                sigmas *= self.density_scale
                                if self.opt.network == "blend4_warp":
                                    warpped_xyzs = density_result_dict["delta_x"]
                                    coords_revert = (warpped_xyzs / (bound - half_grid_size)).clamp(-1., 1.)
                                    coords_revert = (coords_revert + 1.) * (self.grid_size - 1.) / 2.
                                    indices = raymarching.morton3D(coords_revert).long()
                                    # ((warpped_xyzs + 1) * (self.grid_size - 1) / 2).long()
                                # assign 
                                tmp_grid[cas, indices] = sigmas
                
                # dilate the density_grid (less aggressive culling)
                tmp_grid = raymarching.morton3D_dilation(tmp_grid)
                tmp_grid_max_expr = torch.maximum(tmp_grid_max_expr, tmp_grid)

            # ema update
            # tmp_grid = tmp_grid_max_expr # TODO: work的话更换下面的名称
            valid_mask = (self.density_grid >= 0) & (tmp_grid_max_expr >= 0)
            self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid_max_expr[valid_mask])
            self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 regions are viewed as 0 density.
            #self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
            self.iter_density += 1

            # convert to bitfield
            density_thresh = min(self.mean_density, self.density_thresh)
            self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

    def render(self, rays_o, rays_d, rays_expr, bg_coords, poses, staged=False, max_ray_batch=4096,  **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]
        # print(rays_o.shape, rays_d.shape, rays_expr.shape) (1, b, 3) (1, b, 3) (1, 1, 79)

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device


        # never stage when cuda_ray [在不使用cuda_ray的测试时才用到这里]
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            # if kwargs['use_bc']:
            #     bc_temp = kwargs['bc'].contiguous().view(B, -1, 3)
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)

                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], rays_expr, bg_coords[:, head:tail], poses[b:b+1], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image

        else:
            results = _run(rays_o, rays_d, rays_expr, bg_coords, poses, **kwargs)

        return results

