# coding=utf-8
"""
Implementation of the pAdaIN layer. This layer can be added after every convolutional layer and acts
as a regularization which increases overall performance.
It is only applied during training.
"""


import random

import torch
import torch.nn as nn


class PermuteAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, p=0.01, eps=1e-6):
        super(PermuteAdaptiveInstanceNorm2d, self).__init__()
        self.p = p
        self.eps = eps
        self.udistr = torch.distributions.Uniform(0.0, 1.0)
        self.eps = eps

    def forward(self, x):
        # print("执行了P_Adain---------------------------------")
        permute = random.random() < self.p
        if permute and self.training and x.size()[0] == 32:
            perm_indices = torch.randperm(x.size()[0])  # [0,32]随机排列
        elif permute and self.training and x.size()[0] == 64:
            p1 = torch.randperm(int(x.size()[0]/2)).add(32)
            p2 = torch.randperm(int(x.size()[0]/2))
            perm_indices = torch.cat((p2, p1), dim=0)
        else:
            return x
        size = x.size()
        N, C, H, W = size
        if (H, W) == (1, 1):
            print('encountered bad dims')
            return x

        # return adaptive_instance_normalization(x, x[perm_indices], self.eps)
        return style_replacement(x, x[perm_indices], self.eps, self.udistr)

    def extra_repr(self) -> str:
        return 'p={}'.format(
            self.p
        )


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    feat_std = torch.sqrt(feat.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat.detach(), eps)
    content_mean, content_std = calc_mean_std(content_feat, eps)
    content_std = content_std + eps  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def style_replacement(x, style, eps, udistr):

    B = x.size(0)
    C = x.size(1)

    mu = x.mean(dim=[2, 3], keepdim=True)
    var = x.var(dim=[2, 3], keepdim=True)
    sig = (var + eps).sqrt()
    mu, sig = mu.detach(), sig.detach()
    x_normed = (x - mu) / sig  # 风格转换

    mu2 = style.mean(dim=[2, 3], keepdim=True)
    var2 = style.var(dim=[2, 3], keepdim=True)
    sig2 = (var2 + eps).sqrt()

    mu2_gau = torch.randn(mu2.shape).cuda() * 0.1 * (mu2 - mu)
    sig2_gau = torch.randn(sig2.shape).cuda() * 0.1 * (sig2 - sig)
    mu2_p = mu2_gau + mu2
    sig2_p = sig2_gau + sig2

    udistr = udistr.sample((B, C, 1, 1))
    udistr = udistr.to(x.device)
    mu_mix = mu * udistr + mu2_p * (1 - udistr)  # β
    sig_mix = sig * udistr + sig2_p * (1 - udistr)  # γ

    return x_normed * sig_mix + mu_mix