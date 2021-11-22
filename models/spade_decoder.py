# Adapted from https://github.com/NVlabs/SPADE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn import init


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class SPADE_noSPADENorm(nn.Module):
    def __init__(self, norm_type, norm_nc):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=True)
        elif norm_type == 'syncbatch':
            raise NotImplementedError
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=True)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' % norm_type)

    def forward(self, x):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        return normalized


class SPADEResnetBlock_noSPADENorm(nn.Module):
    def __init__(self, fin, fout, use_spectral_norm=False, norm_mode='batch'):
        super().__init__()
        self.norm_mode = norm_mode
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if use_spectral_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        if norm_mode != 'none':
            self.norm_0 = SPADE_noSPADENorm(norm_mode, fin)
            self.norm_1 = SPADE_noSPADENorm(norm_mode, fmiddle)
            if self.learned_shortcut:
                self.norm_s = SPADE_noSPADENorm(norm_mode, fin)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x):
        # seg is texture structure
        x_s = self.shortcut(x)

        if self.norm_mode != 'none':
            dx = self.conv_0(self.actvn(self.norm_0(x)))
            dx = self.conv_1(self.actvn(self.norm_1(dx)))
        else:
            dx = self.conv_0(self.actvn(x))
            dx = self.conv_1(self.actvn(dx))

        out = x_s + dx

        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            if self.norm_mode != 'none':
                x_s = self.conv_s(self.norm_s(x))
            else:
                x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADEGenerator_noSPADENorm(BaseNetwork):
    def __init__(self, nc_init=256, n_upconv=6, nc_out=3, predict_flow=False, upsampling_mode='bilinear',
                 use_spectral_norm=False, norm_mode='batch'):
        super().__init__()

        self.nc_init = nc_init  # should match conv output channels in Encoder(). default: 256
        self.n_upconv = n_upconv
        self.predict_flow = predict_flow
        nf = 16

        self.head_0 = SPADEResnetBlock_noSPADENorm(16 * nf, 16 * nf, use_spectral_norm, norm_mode=norm_mode)

        self.G_middle_0 = SPADEResnetBlock_noSPADENorm(16 * nf, 16 * nf, use_spectral_norm, norm_mode=norm_mode)
        self.G_middle_1 = SPADEResnetBlock_noSPADENorm(16 * nf, 16 * nf, use_spectral_norm, norm_mode=norm_mode)

        self.up_0 = SPADEResnetBlock_noSPADENorm(16 * nf, 8 * nf, use_spectral_norm, norm_mode=norm_mode)
        self.up_1 = SPADEResnetBlock_noSPADENorm(8 * nf, 4 * nf, use_spectral_norm, norm_mode=norm_mode)
        self.up_2 = SPADEResnetBlock_noSPADENorm(4 * nf, 2 * nf, use_spectral_norm, norm_mode=norm_mode)
        self.up_3 = SPADEResnetBlock_noSPADENorm(2 * nf, 1 * nf, use_spectral_norm, norm_mode=norm_mode)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, nc_out, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode=upsampling_mode)

    def forward(self, x):
        x = self.head_0(x)

        x = self.up(x)
        x = self.G_middle_0(x)

        if self.n_upconv >= 6:
            x = self.up(x)

        x = self.G_middle_1(x)

        x = self.up(x)
        x = self.up_0(x)
        x = self.up(x)
        x = self.up_1(x)
        x = self.up(x)
        x = self.up_2(x)
        x = self.up(x)
        x = self.up_3(x)

        if self.n_upconv >= 7:
            x = self.up(x)
            x = self.up_4(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if self.predict_flow:
            x = torch.tanh(x)
        else:
            x = torch.sigmoid(x)
        return x
