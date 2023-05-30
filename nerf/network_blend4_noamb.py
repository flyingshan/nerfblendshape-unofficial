import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
import os

from .gridencoder import ExpGridEncoder
from .modules import MLP
import numpy as np

class MLP2(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=4,
                 hidden_dim=64,
                 geo_feat_dim=64,
                 num_layers_color=1,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 use_latent_code=False,
                 latent_code_dir=None,
                 expr_dim=79, # 
                 level_dim=4,
                 num_levels=16,
                 opt=None,
                 **kwargs,
                 ):
        super().__init__(bound, opt, **kwargs)

        self.opt = opt
        self.use_latent_code = use_latent_code
        self.latent_code_dir = latent_code_dir

        # 初始化latent code, 后续参数直接在load_state_dict的时候会加载进来
        self.init_latent_code()

        # expr basis hashmap

        self.in_dim_expr = expr_dim
        # TODO: num_levels和level_dim是参数
        
        self.encoder = ExpGridEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim, base_resolution=16, 
                                      log2_hashmap_size=16, basis_num=self.in_dim_expr, desired_resolution=2048) # 1024*self.bound

        ambient_dim = 2
        self.encoder_ambient, self.in_dim_ambient = get_encoder('tiledgrid', input_dim=ambient_dim, 
                                      num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048)
        self.ambient_net = MLP(in_ch=num_levels * level_dim + self.in_dim_expr, out_ch=2, width=64, depth=3)

        # sigma network
        self.in_dim = num_levels * level_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        # self.encoder, self.in_dim = get_encoder(encoding, num_levels=16, level_dim=4, base_resolution=16, log2_hashmap_size=14, desired_resolution=1024)

        self.sigma_net = MLP(in_ch=self.in_dim + self.opt.ind_dim_head if use_latent_code else self.in_dim + self.in_dim_ambient, 
                             out_ch=1+self.geo_feat_dim, 
                             depth=self.num_layers, 
                             width=self.hidden_dim, 
                             output_activation=None, 
                             use_bias=False)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        self.color_net = MLP(in_ch=self.in_dim_dir+self.geo_feat_dim, 
                             out_ch=3,
                             depth=self.num_layers_color, 
                             width=self.hidden_dim_color,
                             output_activation=None, 
                             use_bias=False)

        # background network
        self.bg_net = None

        if self.torso:
            # torso deform network
            self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('frequency', input_dim=2, multires=10)
            self.pose_encoder, self.pose_in_dim = get_encoder('frequency', input_dim=6, multires=4) 
            self.torso_deform_net = MLP2(self.torso_deform_in_dim + self.pose_in_dim + self.opt.ind_dim_torso, 2, 64, 3)

            # torso color network
            self.torso_encoder, self.torso_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048)
            self.torso_net = MLP2(self.torso_in_dim + self.torso_deform_in_dim + self.pose_in_dim + self.opt.ind_dim_torso, 4, 32, 3)

    def forward_torso(self, x, poses, enc_a, **kwargs):
        # x: [N, 2] in [-1, 1]
        # head poses: [1, 6]
        # c: [1, ind_dim], individual code

        # test: shrink x
        
        if kwargs['mode'] == 'train':
            index = kwargs['index']
            c = self.individual_codes_torso[index].repeat(x.shape[0], 1)
        else:
            c = self.individual_codes_torso[0].repeat(x.shape[0], 1)

        x = x * self.opt.torso_shrink

        # deformation-based 
        enc_pose = self.pose_encoder(poses)
        enc_x = self.torso_deform_encoder(x)

        # if c is not None:
        h = torch.cat([enc_x, enc_pose.repeat(x.shape[0], 1), c], dim=-1)

        dx = self.torso_deform_net(h)

        x = (x + dx).clamp(-1, 1)

        x = self.torso_encoder(x, bound=1)

        # h = torch.cat([x, h, enc_a.repeat(x.shape[0], 1)], dim=-1)
        h = torch.cat([x, h], dim=-1)

        h = self.torso_net(h)

        alpha = torch.sigmoid(h[..., :1])
        color = torch.sigmoid(h[..., 1:])

        return alpha, color, dx

    def forward(self, x, d, expr, **kwargs):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        if kwargs['use_latent_code']:
            index = kwargs['index']
            if kwargs['mode'] == 'train':
                latent_code = self.latent_codes[index].repeat(x.shape[0], 1)
            else:
                latent_code = self.latent_codes[0].repeat(x.shape[0], 1)
        # print(self.expr_min.shape, expr.shape)

        # if kwargs['mode'] != 'train':
        expr = torch.minimum(self.expr_max, expr)
        expr = torch.maximum(self.expr_min, expr)

        h = self.encoder(x, expr, bound=self.bound)

        # print(expr.shape)
        expr = expr.repeat(x.shape[0], 1) 
        ambient = torch.cat([h, expr], dim=1)
        ambient = self.ambient_net(ambient)
        ambient = torch.tanh(ambient).float()

        if kwargs['use_latent_code']:
            h = torch.cat([h,latent_code], dim = -1)
        else:
            h = torch.cat([h], dim = -1)

        h = self.sigma_net(h)

        # sigma = F.softplus(h[..., 0])
        sigma = trunc_exp(h[..., 0])

        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        h = h.float()

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        latent_code_res = None
        if kwargs['use_latent_code']:
            latent_code_res = latent_code[0,:] # 用于计算loss

        result_dict = {
            'sigma': sigma,
            'color': color,
            'latent_code_res': latent_code_res,
            'ambient': ambient, 
            'latent_coef':  None,
        }

        return result_dict

    def density(self, x , expr, **kwargs):
        # x: [N, 3], in [-bound, bound]
        if kwargs['use_latent_code']:
            index = kwargs['index']
            if kwargs['mode'] == 'train':
                latent_code = self.latent_codes[index].repeat(x.shape[0], 1)
            else:
                latent_code = self.latent_codes[0].repeat(x.shape[0], 1)

        # if kwargs['mode'] != 'train':
        expr = torch.minimum(self.expr_max, expr)
        expr = torch.maximum(self.expr_min, expr)

        # print(expr.shape)
        h = self.encoder(x, expr, bound=self.bound)

        expr = expr.repeat(x.shape[0], 1) 
        ambient = torch.cat([h, expr], dim=1)
        ambient = self.ambient_net(ambient)
        ambient = torch.tanh(ambient).float()

        if kwargs['use_latent_code']:
            h = torch.cat([h,latent_code], dim = -1)
        else:
            h = torch.cat([h], dim = -1)

        h = self.sigma_net(h)
        # sigma = F.softplus(h[..., 0])        
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # print(sigma.shape, geo_feat.shape)
        result_dict =   {
            'sigma': sigma,
            'geo_feat': geo_feat
        }

        if kwargs['use_latent_code']:
            result_dict['latent_code'] = latent_code[0,:] # 用于计算loss
        
        # print(result_dict.keys())
        return result_dict

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]


        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        h = h.float()
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils

    def init_latent_code(self):
        if self.opt.torso:
            self.individual_codes_torso = nn.Parameter(torch.randn(10000, self.opt.ind_dim_torso).cuda() * 0.1)
        if self.use_latent_code:
            self.latent_codes = nn.Parameter(torch.rand(10000, self.opt.ind_dim_head).cuda() * 0.1)

    def get_params(self, lr, loader, device):

        if self.torso:
            params = [
                {'params': self.torso_encoder.parameters(), 'lr': 5e-3},
                {'params': self.torso_net.parameters(), 'lr': 5e-4},
                {'params': self.torso_deform_net.parameters(), 'lr': 5e-4},
            ]
            params.append({'params': self.individual_codes_torso, 'lr': 5e-4})
            return params

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.encoder_ambient.parameters(), 'lr': lr}, 
            {'params': self.ambient_net.parameters(), 'lr': lr}, 
        ]

        if self.use_latent_code:
            params.append({'params': self.latent_codes, 'lr': lr})

        return params

    def save_latent_codes(self, save_dir="./checkpoints"):
        save_dict = {"latent_codes": self.latent_codes}
        torch.save(save_dict, os.path.join(save_dir, "latent_codes.ckpt"))
        if self.opt.torso:
            save_dict = {"latent_codes_torso": self.individual_codes_torso}
            torch.save(save_dict, os.path.join(save_dir, "latent_codes_torso.ckpt"))

    def load_min_max_expr(self, root_path, device):
        self.expr_max = torch.from_numpy(np.load(os.path.join(root_path, 'expr_max.npy'))).to(device).unsqueeze(0) # 1, expr_dim
        self.expr_min = torch.from_numpy(np.load(os.path.join(root_path, 'expr_min.npy'))).to(device).unsqueeze(0)
        print("load expr_max successfully:expr_max is :")
        print(self.expr_max)
        print("load expr_min successfully:expr_min is :")
        print(self.expr_min)
