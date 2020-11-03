from __future__ import division

import torch
import torch.nn as nn
import torchfile

from modelsNIPS import decoder1, decoder2, decoder3, decoder4, decoder5
from modelsNIPS import encoder1, encoder2, encoder3, encoder4, encoder5


class WCT(nn.Module):
    def __init__(self, args):
        super(WCT, self).__init__()
        # load pre-trained network
        vgg1 = torchfile.load(args.vgg1)
        decoder1_torch = torchfile.load(args.decoder1)
        vgg2 = torchfile.load(args.vgg2)
        decoder2_torch = torchfile.load(args.decoder2)
        vgg3 = torchfile.load(args.vgg3)
        decoder3_torch = torchfile.load(args.decoder3)
        vgg4 = torchfile.load(args.vgg4)
        decoder4_torch = torchfile.load(args.decoder4)
        vgg5 = torchfile.load(args.vgg5)
        decoder5_torch = torchfile.load(args.decoder5)

        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)
        self.e2 = encoder2(vgg2)
        self.d2 = decoder2(decoder2_torch)
        self.e3 = encoder3(vgg3)
        self.d3 = decoder3(decoder3_torch)
        self.e4 = encoder4(vgg4)
        self.d4 = decoder4(decoder4_torch)
        self.e5 = encoder5(vgg5)
        self.d5 = decoder5(decoder5_torch)

    def whiten_and_color(self, f_c_v, f_s_v):
        f_c_size = f_c_v.size()
        c_mean = torch.mean(f_c_v, 1)  # c x (w x h)
        c_mean = c_mean.unsqueeze(1).expand_as(f_c_v)
        # 均值归一化
        f_c_v = f_c_v - c_mean
        # 构造协方差矩阵 f_c_size[0] = channel, f_c_size[1] = width * height
        content_conv = torch.mm(f_c_v, f_c_v.t()).div(f_c_size[1] - 1)  # + torch.eye(f_c_size[0]).double()
        # 计算所有奇异值
        c_u, c_e, c_v = torch.svd(content_conv, some=False)  # n*n, n*m, m*m

        # 获取大于 0.00001 的特征值个数，特征值从大到小排列
        k_c = f_c_size[0]
        for i in range(f_c_size[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        d_c = (c_e[0:k_c]).pow(-0.5)  # D_c^(-1/2) 协方差矩阵特征值的对角阵的 -1/2 次
        e_c = c_v[:, 0:k_c]  # E_c 特征值对应的正交矩阵
        whiten_f_c = torch.mm(torch.mm(torch.mm(e_c, torch.diag(d_c)), e_c.t()), f_c_v)  # 目标内容 f^hat_c

        f_s_size = f_s_v.size()
        s_mean = torch.mean(f_s_v, 1)
        f_s_v = f_s_v - s_mean.unsqueeze(1).expand_as(f_s_v)
        style_conv = torch.mm(f_s_v, f_s_v.t()).div(f_s_size[1] - 1)
        s_u, s_e, s_v = torch.svd(style_conv, some=False)

        s_k = f_s_size[0]
        for i in range(f_s_size[0]):
            if s_e[i] < 0.00001:
                s_k = i
                break

        d_s = (s_e[0:s_k]).pow(0.5)
        e_s = s_v[:, 0:s_k]
        target_feature = torch.mm(torch.mm(torch.mm(e_s, torch.diag(d_s)), e_s.t()), whiten_f_c)
        target_feature = target_feature + s_mean.unsqueeze(1).expand_as(target_feature)
        return target_feature

    def transform(self, f_c, f_s, f_cs, alpha):
        f_c = f_c.double()
        f_s = f_s.double()
        c_channel, c_width, c_height = f_c.size(0), f_c.size(1), f_c.size(2)
        _, s_width, s_height = f_s.size(0), f_s.size(1), f_s.size(2)
        f_c_view = f_c.view(c_channel, -1)
        f_s_view = f_s.view(c_channel, -1)

        target_feature = self.whiten_and_color(f_c_view, f_s_view)
        target_feature = target_feature.view_as(f_c)
        # 合成目标图像
        f_cs_des = alpha * target_feature + (1.0 - alpha) * f_c
        f_cs_des = f_cs_des.float().unsqueeze(0)
        with torch.no_grad():
            f_cs.resize_(f_cs_des.size()).copy_(f_cs_des)
        return f_cs
