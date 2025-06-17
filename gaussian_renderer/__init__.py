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
from typing import Union
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh
import gsplat


def render(
        viewpoint_camera,
        pc: Union[GaussianModel, FlameGaussianModel],
        pipe,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
    ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    render, alpha, info = gsplat.rasterization(
        means=means3D,
        quats=rotations,
        scales=scales*scaling_modifier,
        opacities=opacity.squeeze(-1),
        colors=colors_precomp if colors_precomp is not None else shs,
        viewmats=viewpoint_camera.world_view_transform.cuda().transpose(0, 1).unsqueeze(0),
        Ks=viewpoint_camera.K.cuda().unsqueeze(0),
        width=viewpoint_camera.image_width,
        height=viewpoint_camera.image_height,
        sh_degree=pc.active_sh_degree,
        backgrounds=bg_color,
    )
    render = render.squeeze(0).permute(2, 0, 1)
    alpha = alpha.squeeze(0).permute(2, 0, 1)

    radii = torch.zeros(means3D.shape[0], device=info["radii"].device, dtype=info["radii"].dtype)
    radii[info["gaussian_ids"]] = info["radii"].max(dim=1).values
    return {
        "render": render,
        "alpha": alpha,
        "radii": radii,
        "info": info,
    }
