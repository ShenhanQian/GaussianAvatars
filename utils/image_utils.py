#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from matplotlib import cm

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def error_map(img1, img2):
    error = (img1 - img2).mean(dim=0) / 2 + 0.5
    cmap = cm.get_cmap("seismic")
    error_map = cmap(error.cpu())
    return torch.from_numpy(error_map[..., :3]).permute(2, 0, 1)
