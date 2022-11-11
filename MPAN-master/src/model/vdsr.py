from model import common

import torch

from math import sqrt

import torch.nn as nn
import torch.nn.init as init

url = {
    'r20f64': ''
}

def make_model(args, parent=False):
    return VDSR(args)

# class VDSR(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(VDSR, self).__init__()
#
#         n_resblocks = args.n_resblocks
#         n_feats = args.n_feats
#         scale = args.scale[0]
#         kernel_size = 3
#         self.url = url['r{}f{}'.format(n_resblocks, n_feats)]
#         self.sub_mean = common.MeanShift(args.rgb_range)
#         self.add_mean = common.MeanShift(args.rgb_range, sign=1)
#
#         def basic_block(in_channels, out_channels, act):
#             return common.BasicBlock(
#                 conv, in_channels, out_channels, kernel_size,
#                 bias=True, bn=False, act=act
#             )
#
#         # define body module
#         m_body = []
#         m_body.append(basic_block(args.n_colors, n_feats, nn.ReLU(True)))
#         for _ in range(n_resblocks - 2):
#             m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
#         m_body.append(basic_block(n_feats, args.n_colors, None))
#
#         m_tail = [
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
#             common.Upsampler(conv, scale, n_feats, act=False),
#             #conv(n_feats, args.n_colors, kernel_size)
#         ]
#
#         self.body = nn.Sequential(*m_body)
#         self.tail = nn.Sequential(*m_tail)
#
#     def forward(self, x):
#         x = self.sub_mean(x)
#         #print(x.shape)
#         res = self.body(x)
#         res += x
#         print(res.shape)
#         #x = self.tail(res)
#         up = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
#         res = up(res)
#         print(res.shape)
#         x = self.add_mean(res)
#         #print(x.shape)
#
#         return x

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        #up = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        #out = up(out)
        #print(out.shape)
        return out