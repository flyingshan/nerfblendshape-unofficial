import torch
import torch.nn as nn
import numpy as np
import functools

from encoding import get_encoder

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x

class MLP(nn.Module):
    """A multi-layer perceptron.
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        hidden_channels: The number of hidden channels.
        hidden_layers: The number of hidden layers.
        hidden_activation: The activation function for the hidden layers.
        hidden_norm: A string indicating the type of norm to use for the hidden
            layers.
        out_activation: The activation function for the output layer.
        out_norm: A string indicating the type of norm to use for the output
            layer.
        dropout: The dropout rate.
    """

    def __init__(self,in_ch:int,out_ch:int, depth:int=8,width:int=256,hidden_init=None,hidden_activation=None,
            hidden_norm=None,output_init=None, output_activation=None,
            use_bias=True,skips=None):
        super(MLP, self).__init__() 
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.depth = depth
        self.width = width
        if hidden_init is None:
            self.hidden_init = nn.init.xavier_uniform_
        else:
            self.hidden_init = hidden_init

        if hidden_activation is None:
            self.hidden_activation = nn.ReLU()
        else:
            self.hidden_activation = hidden_activation

        self.hidden_norm = hidden_norm

        if output_init is None:
            self.output_init = nn.init.xavier_uniform_
        else:
            self.output_init = output_init

        if output_activation == None:
            self.output_activation = nn.Identity()
        else:
            self.output_activation = output_activation

        # self.use_bias = use_bias
        if skips is None:
            self.skips = [4,]
        else:
            self.skips = skips

        self.linears = nn.ModuleList([nn.Linear(in_ch, width, bias=use_bias)] + 
            [nn.Linear(width, width, bias=use_bias) if i not in self.skips else 
            nn.Linear(width+ in_ch, width, bias=use_bias) for i in range(depth-1)])
        self.logit_layer = nn.Linear(width, out_ch, bias=use_bias)

        # initalize using glorot
        for _, linear in enumerate(self.linears):
            self.hidden_init(linear.weight)
        # initialize output layer
        if self.output_init is not None:
            self.output_init(self.logit_layer.weight)

        if self.hidden_norm is not None:
            #TODO
            pass
            # self.norm_layers = nn.ModuleList([get_norm_layer(self.hidden_norm) for _ in range(depth)])

    def forward(self,inputs):
        x = inputs
        # print(x[:3], '1111')
        for i, linear in enumerate(self.linears):
            x = linear(x)
            x = self.hidden_activation(x)
            # if self.hidden_norm is not None:
            #     x = self.norm_layers[i](x)
            if i in self.skips:
                x = torch.cat([x,inputs],-1)
        # print(x[:3], '2222')
        x = self.logit_layer(x)
        x = self.output_activation(x)
        return x


class SliceMLP(nn.Module):
    def __init__(self,in_ch:int=3, in_ch_embed:int=8, out_ch:int=3, depth:int=6,
                    width:int=64, skips=None, use_residual=False, use_bias=False):
        super(SliceMLP, self).__init__()
        self.out_ch = out_ch
        self.depth = depth
        self.width = width
        self.in_ch_embed = in_ch_embed # default is 8 according to the paper
        self.use_bias = use_bias
        # self.bound = bound
        # assume use identity
        """
        self.n_freq = 7 #TODO hardcoded
        self.in_ch = model_utils.get_posenc_ch_orig(in_ch,self.n_freq) + in_ch_embed
        """
        self.encoder, self.pos_embed_dim = get_encoder(encoding="frequency", multires=6)
        self.in_ch = self.pos_embed_dim + in_ch_embed

        if skips is None:
            self.skips = [3,]
        else:
            self.skips = skips

        self.hidden_init = nn.init.xavier_uniform_
        self.output_init = functools.partial(nn.init.normal_,std=1e-5)
        self.use_residual = use_residual
        self.mlp = MLP(in_ch=self.in_ch,
                     out_ch=self.out_ch,
                     depth=self.depth,
                     hidden_init = self.hidden_init, 
                     output_init = self.output_init,
                     width=self.width,
                     skips=self.skips,
                     use_bias=self.use_bias)


    def forward(self,pts,embed,alpha = None):
        """
        points_feat = model_utils.posenc_orig(pts,self.n_freq)
        """
        points_feat = self.encoder(pts) # frequency所以不需要bound参数
        inputs = torch.cat([points_feat,embed],dim=-1)
        if self.use_residual:
            return self.mlp(inputs) + embed
        else:
            return self.mlp(inputs)