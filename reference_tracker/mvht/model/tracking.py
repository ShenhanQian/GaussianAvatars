# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from mvht.config.base import BaseTrackingConfig
from mvht.model.flame import FlameHead, FlameTexPCA, FlameTexPainted, FlameMask, FlameUvMask
from mvht.model.lbs import batch_rodrigues
from mvht.util.general import blur_tensor_adaptive, DecayScheduler
from mvht.util.graphics import (
    get_mtl_content,
    get_obj_content,
    normalize_image_points,
)
from mvht.util.log import get_logger
from mvht.util.visualization import plot_landmarks_2d
from mvht.data.multi_view_head_dataset import MultiViewHeadDataset

from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Literal
from functools import partial
import tyro
import yaml
from datetime import datetime
import threading
from typing import Optional
import time
import os

logger = get_logger(__name__)


def to_batch(x, indices):
    return torch.stack([x[i] for i in indices])


class FlameTracker:
    def __init__(self, cfg: BaseTrackingConfig):
        self.cfg = cfg
        
        self.device = cfg.device
        self.tb_writer = None

        # model
        self.flame = FlameHead(
            cfg.model.n_shape, 
            cfg.model.n_expr, 
            add_teeth=cfg.model.add_teeth,
            remove_lip_inside=cfg.model.remove_lip_inside,
            face_clusters=cfg.model.tex_clusters,
            ).to(self.device)

        if cfg.model.tex_painted:
            self.flame_tex_painted = FlameTexPainted(tex_size=cfg.model.tex_resolution).to(self.device)
        else:
            self.flame_tex_pca = FlameTexPCA(cfg.model.n_tex, tex_size=cfg.model.tex_resolution).to(self.device)

        self.flame_uvmask = FlameUvMask().to(self.device)

        # renderer for visualization, dense photometric energy
        if self.cfg.render.backend == 'nvdiffrast':
            from mvht.util.render_nvdiffrast import NVDiffRenderer

            self.render = NVDiffRenderer(
                use_opengl=self.cfg.render.use_opengl,
                lighting_type=self.cfg.render.lighting_type,
                lighting_space=self.cfg.render.lighting_space,
                disturb_rate_fg=self.cfg.render.disturb_rate_fg,
                disturb_rate_bg=self.cfg.render.disturb_rate_bg,
                fid2cid=self.flame.mask.fid2cid,
            )
        elif self.cfg.render.backend == 'pytorch3d':
            from mvht.util.render_pytorch3d import PyTorch3DRenderer

            self.render = PyTorch3DRenderer()
        else:
            raise NotImplementedError(f"Unknown renderer backend: {self.cfg.render.backend}")
    
    def load_from_tracked_flame_params(self, fp):
        """
        loads checkpoint from tracked_flame_params file. Counterpart to export_result()
        :param fp:
        :return:
        """
        report = np.load(fp)

        # LOADING PARAMETERS
        def load_param(param, ckpt_array):
            param.data[:] = torch.from_numpy(ckpt_array).to(param.device)

        def load_param_list(param_list, ckpt_array):
            for i in range(min(len(param_list), len(ckpt_array))):
                load_param(param_list[i], ckpt_array[i])

        load_param_list(self.rotation, report["rotation"])
        load_param_list(self.translation, report["translation"])
        load_param_list(self.neck_pose, report["neck_pose"])
        load_param_list(self.jaw_pose, report["jaw_pose"])
        load_param_list(self.eyes_pose, report["eyes_pose"])
        if self.flame.has_neck_base_joint:
            load_param_list(self.neck_base_pose, report["neck_base_pose"])
        load_param(self.shape, report["shape"])
        load_param_list(self.expr, report["expr"])
        load_param(self.lights, report["lights"])
        # self.frame_idx = report["n_processed_frames"]
        if not self.calibrated:
            load_param(self.K, report["K"])
            load_param(self.RT, report["RT"])
        
        if not self.cfg.model.tex_painted:
            if "tex" in report:
                load_param(self.tex_pca, report["tex"])
            else:
                logger.warn("No tex_extra found in flame_params!")
        
        if self.cfg.model.tex_extra:
            if "tex_extra" in report:
                load_param(self.tex_extra, report["tex_extra"])
            else:
                logger.warn("No tex_extra found in flame_params!")
        
        if self.cfg.model.use_static_offset:
            if "static_offset" in report:
                load_param(self.static_offset, report["static_offset"])
            else:
                logger.warn("No static_offset found in flame_params!")

        if self.cfg.model.use_dynamic_offset:
            if "dynamic_offset" in report:
                load_param_list(self.dynamic_offset, report["dynamic_offset"])
            else:
                logger.warn("No dynamic_offset found in flame_params!")

    def trimmed_decays(self, is_init):
        decays = {}
        for k, v in self.decays.items():
            if is_init and "init" in k or not is_init and "init" not in k:
                decays[k.replace("_init", "")] = v
        return decays

    def clear_cache(self):
        self.render.clear_cache()

    def get_current_frame(self, frame_idx, include_keyframes=False):
        """
        Creates a single item batch from the frame data at index frame_idx in the dataset.
        If include_keyframes option is set, keyframe data will be appended to the batch. However,
        it is guaranteed that the frame data belonging to frame_idx is at position 0
        :param frame_idx:
        :return:
        """
        indices = [frame_idx]
        if include_keyframes:
            indices += self.cfg.exp.keyframes

        samples = []
        for idx in indices:
            sample = self.dataset.getitem_by_timestep(idx)
            # sample["timestep_index"] = idx

            # for k, v in sample.items():
            #     if isinstance(v, torch.Tensor):
            #         sample[k] = v[None, ...].to(self.device)

            samples.append(sample)

        # if also keyframes have been loaded, stack all data
        sample = {}
        for k, v in samples[0].items():
            values = [s[k] for s in samples]
            if isinstance(v, torch.Tensor):
                values = torch.cat(values, dim=0)
            sample[k] = values

        if "lmk2d_iris" in sample:
            sample["lmk2d"] = torch.cat([sample["lmk2d"], sample["lmk2d_iris"]], dim=1)
        return sample

    def fill_cam_params_into_sample(self, sample):
        """
        Adds intrinsics and extrinics to sample, if data is not calibrated
        """
        if self.calibrated:
            assert "intrinsic" in sample
            assert "extrinsic" in sample
        else:
            b, _, h, w = sample["rgb"].shape
            # K = torch.eye(3, 3).to(self.device)

            # denormalize cam params
            f = self.K * max(h, w)
            # cx, cy = self.K[[1]] * w, self.K[[2]] * h
            cx, cy = 0.5 * torch.ones_like(self.K) * w, 0.5 * torch.ones_like(self.K) * h

            sample["intrinsic"] = torch.stack([f, f, cx, cy], dim=1)
            sample["extrinsic"] = self.RT[None, ...].expand(b, -1, -1)

    def configure_optimizer(self, params, lr_scale=1.0):
        """
        Creates optimizer for the given set of parameters
        :param params:
        :return:
        """
        # copy dict because we will call 'pop'
        params = params.copy()
        param_groups = []
        default_lr = self.cfg.lr.base

        # dict map group name to param dict keys
        group_def = {
            "translation": ["translation"],
            "expr": ["expr"],
            "light": ["lights"],
        }
        if not self.calibrated:
            group_def ["cam"] = ["cam"]
        if self.cfg.model.use_static_offset:
            group_def ["static_offset"] = ["static_offset"]
        if self.cfg.model.use_dynamic_offset:
            group_def ["dynamic_offset"] = ["dynamic_offset"]

        # dict map group name to lr
        group_lr = {
            "translation": self.cfg.lr.translation,
            "expr": self.cfg.lr.expr,
            "light": self.cfg.lr.light,
        }
        if not self.calibrated:
            group_lr["cam"] = self.cfg.lr.camera
        if self.cfg.model.use_static_offset:
            group_lr["static_offset"] = self.cfg.lr.static_offset
        if self.cfg.model.use_dynamic_offset:
            group_lr["dynamic_offset"] = self.cfg.lr.dynamic_offset

        for group_name, param_keys in group_def.items():
            selected = []
            for p in param_keys:
                if p in params:
                    selected += params.pop(p)
            if len(selected) > 0:
                param_groups.append({"params": selected, "lr": group_lr[group_name] * lr_scale})

        # create default group with remaining params
        selected = []
        for _, v in params.items():
            selected += v
        param_groups.append({"params": selected})

        optim = torch.optim.Adam(param_groups, lr=default_lr * lr_scale)
        return optim

    def initialize_frame(self, frame_idx):
        """
        Initializes parameters of frame frame_idx
        :param frame_idx:
        :return:
        """
        if frame_idx > 0:
            self.initialize_from_previous(frame_idx)

    def initialize_from_previous(self, frame_idx):
        """
        Initializes the flame parameters with the optimized ones from the previous frame
        :param frame_idx:
        :return:
        """
        if frame_idx == 0:
            return

        param_list = [
            self.expr,
            self.neck_pose,
            self.jaw_pose,
            self.translation,
            self.rotation,
            self.eyes_pose,
        ]
        if self.flame.has_neck_base_joint:
            param_list.append(self.neck_base_pose)

        for param in param_list:
            param[frame_idx].data = param[frame_idx - 1].detach().clone().data

    def select_frame_indices(self, frame_idx, include_keyframes):
        indices = [frame_idx]
        if include_keyframes:
            indices += self.cfg.exp.keyframes
        return indices

    def forward_flame(self, frame_idx, include_keyframes):
        """
        Evaluates the flame model using the given parameters
        :param flame_params:
        :return:
        """
        indices = self.select_frame_indices(frame_idx, include_keyframes)

        dynamic_offset = to_batch(self.dynamic_offset, indices) if self.cfg.model.use_dynamic_offset else None
        neck_base_pose = to_batch(self.neck_base_pose, indices) if self.flame.has_neck_base_joint else None

        ret = self.flame(
            self.shape[None, ...].expand(len(indices), -1),
            to_batch(self.expr, indices),
            to_batch(self.rotation, indices),
            to_batch(self.neck_pose, indices),
            to_batch(self.jaw_pose, indices),
            to_batch(self.eyes_pose, indices),
            to_batch(self.translation, indices),
            return_verts_cano=True,
            static_offset=self.static_offset,
            dynamic_offset=dynamic_offset,
            neck_base=neck_base_pose,
        )
        verts, verts_cano, lmks = ret[0], ret[1], ret[2]
        albedos = self.get_albedo().expand(len(indices), -1, -1, -1)
        return verts, verts_cano, lmks, albedos
    
    def get_base_texture(self):
        if self.cfg.model.tex_extra and not self.cfg.model.residual_tex:
            albedos_base = self.tex_extra[None, ...]
        else:
            if self.cfg.model.tex_painted:
                albedos_base = self.flame_tex_painted()
            else:
                albedos_base = self.flame_tex_pca(self.tex_pca[None, :])

        return albedos_base
    
    def get_albedo(self):
        albedos_base = self.get_base_texture()

        if self.cfg.model.tex_extra and self.cfg.model.residual_tex:
            albedos_res = self.tex_extra[None, :]
            if albedos_base.shape[-1] != albedos_res.shape[-1] or albedos_base.shape[-2] != albedos_res.shape[-2]:
                albedos_base = F.interpolate(albedos_base, albedos_res.shape[-2:], mode='bilinear')
            albedos = albedos_base + albedos_res
        else:
            albedos = albedos_base

        return albedos

    def rasterize_flame(
        self, sample, verts, faces, camera_index=None, train_mode=False
    ):

        """
        Rasterizes the flame head mesh
        :param verts:
        :param albedos:
        :param K:
        :param RT:
        :param resolution:
        :param scale:
        :param only_face:
        :param use_cache:
        :return:
        """
        if not train_mode:
            use_cache = False
            scale = 1
            require_grad = False
        else:
            if self.cfg.render.diff_rast:
                use_cache = False
                scale = 1
                require_grad = True
            else:
                use_cache = True
                scale=self.cfg.render.sampling_scale
                require_grad = False

        # cameras parameters
        K = sample["intrinsic"].clone().to(self.device)
        K[:, :2] *= scale
        RT = sample["extrinsic"].to(self.device)
        if camera_index is not None:
            K = K[[camera_index]]
            RT = RT[[camera_index]]

        H, W = self.image_size
        image_size = int(H * scale), int(W * scale)
        
        # rasterize fragments
        rast_dict = self.render.rasterize(verts, faces, RT, K, image_size, use_cache, require_grad)
        return rast_dict

    @torch.no_grad()
    def get_background_color(self, gt_rgb, gt_alpha, stage):
        if stage is None:  # when stage is None, it means we are in the evaluation mode
            background = self.cfg.render.background_eval
        else:
            background = self.cfg.render.background_train

        if background == 'target':
            """use gt_rgb as background"""
            color = gt_rgb.permute(0, 2, 3, 1)
        elif background == 'white':
            color = [1, 1, 1]
        elif background == 'black':
            color = [0, 0, 0]
        else:
            raise NotImplementedError(f"Unknown background mode: {background}")
        return color
    
    def render_rgba(
            self, rast_dict, verts, faces, albedos, lights, background_color=[1, 1, 1],
            align_texture_except_fid=None, align_boundary_except_vid=None,
        ):
        """
        Renders the rgba image from the rasterization result and
        the optimized texture + lights
        """
        faces_uv = self.flame.textures_idx
        if self.cfg.render.backend == 'nvdiffrast':
            verts_uv = self.flame.verts_uvs.clone()
            verts_uv[:, 1] = 1 - verts_uv[:, 1]
            tex = albedos

            render_out = self.render.render_rgba(
                rast_dict, verts, faces, verts_uv, faces_uv, tex, lights, background_color,
                align_texture_except_fid, align_boundary_except_vid,
            )
            render_out = {k: v.permute(0, 3, 1, 2) for k, v in render_out.items()}
        elif self.cfg.render.backend == 'pytorch3d':
            B = verts.shape[0]  # TODO: double check
            verts_uv = self.flame.face_uvcoords.repeat(B, 1, 1)
            tex = albedos.expand(B, -1, -1, -1)

            rgba = self.render.render_rgba(
                rast_dict, verts, faces, verts_uv, faces_uv, tex, lights, background_color
            )
            render_out = {'rgba': rgba.permute(0, 3, 1, 2)}
        else:
            raise NotImplementedError(f"Unknown renderer backend: {self.cfg.render.backend}")
        
        return render_out

    def render_normal(self, rast_dict, verts, faces):
        """
        Renders the rgba image from the rasterization result and
        the optimized texture + lights
        """
        uv_coords = self.flame.face_uvcoords
        uv_coords = uv_coords.repeat(verts.shape[0], 1, 1)
        return self.render.render_normal(rast_dict, verts, faces, uv_coords)

    def compute_lmk_energy(self, sample, pred_lmks):
        """
        Computes the landmark energy loss term between groundtruth landmarks and flame landmarks
        :param sample:
        :param pred_lmks:
        :return: the lmk loss for all 68 facial landmarks, a separate 2 pupil landmark loss and
                 a relative eye close term
        """
        img_size = sample["rgb"].shape[-2:]

        # ground-truth landmark
        lmk2d = sample["lmk2d"].clone().to(pred_lmks)
        lmk2d, confidence = lmk2d[:, :, :2], lmk2d[:, :, 2]
        lmk2d[:, :, 0], lmk2d[:, :, 1] = normalize_image_points(
            lmk2d[:, :, 0], lmk2d[:, :, 1], img_size
        )

        # predicted landmark
        K = sample["intrinsic"].to(self.device)
        RT = sample["extrinsic"].to(self.device)
        pred_lmk_ndc = self.render.world_to_ndc(pred_lmks, RT, K, img_size, flip_y=True)
        pred_lmk2d = pred_lmk_ndc[:, :, :2]

        if self.cfg.w.disable_boundary_landmarks:
            diff = lmk2d[:, 17:68] - pred_lmk2d[:, 17:68]
            confidence = confidence[:, 17:68]
        else:
            diff = lmk2d[:, :68] - pred_lmk2d[:, :68]
            confidence = confidence[:, :68]

        # compute general landmark term
        lmk_loss = torch.norm(diff, dim=2, p=1) * confidence

        result_dict = {
            "gt_lmk2d": lmk2d,
            "pred_lmk2d": pred_lmk2d,
        }

        return lmk_loss.mean(), result_dict

    def compute_photometric_energy(
        self,
        sample,
        verts,
        faces,
        albedos,
        rast_dict,
        step_i=None,
        stage=None,
        include_keyframes=False,
    ):
        """
        Computes the dense photometric energy
        :param sample:
        :param vertices:
        :param albedos:
        :return:
        """
        gt_rgb = sample["rgb"].to(albedos)
        gt_alpha = sample["alpha_map"].to(albedos)
            
        lights = self.lights[None] if self.lights is not None else None
        bg_color = self.get_background_color(gt_rgb, gt_alpha, stage)
        render_out = self.render_rgba(rast_dict, verts, faces, albedos, lights, bg_color)
        pred_rgb = render_out['rgba'][:, :3]
        pred_alpha = render_out['rgba'][:, 3:]
        pred_mask = render_out['rgba'][:, [3]].detach() > 0
        pred_mask = pred_mask.expand(-1, 3, -1, -1)

        results_dict = {}

        # decays = self.trimmed_decays(sample["timestep_index"][0] == 0)
        screen_coords = rast_dict["screen_coords"]
        # ---- rgb loss ----
        with torch.no_grad():
            if step_i is not None:
                # coarse to fine
                decayed_sigma = decays["blur_sigma"].get(step_i)

                gt_rgb = blur_tensor_adaptive(gt_rgb, sigma=decayed_sigma)[0]
                # pred_rgb = blur_tensor_adaptive(pred_rgb, sigma=decayed_sigma)[0]

        screen_colors = F.grid_sample(gt_rgb, screen_coords)
        error_rgb = torch.where(
            pred_mask,
            (screen_colors - pred_rgb).abs(),
            torch.zeros_like(pred_rgb),
        )
        color_loss = error_rgb.sum() / pred_mask.detach().sum()

        results_dict.update(
            {
                "gt_rgb": gt_rgb,
                # "gt_rgb": screen_colors,
                "pred_rgb": pred_rgb,
                "error_rgb": error_rgb,
                "pred_alpha": pred_alpha,
            }
        )

        # ---- silhouette loss ----
        with torch.no_grad():
            if step_i is not None:
                # coarse to fine
                sigma = 3

                gt_alpha = blur_tensor_adaptive(gt_alpha, sigma=sigma)[0]
                pred_alpha_blurred = blur_tensor_adaptive(pred_alpha, sigma=sigma)[0]

        screen_alpha = F.grid_sample(gt_alpha, screen_coords)
        error_alpha = torch.where(
            pred_mask,
            (screen_alpha - pred_alpha_blurred).abs(),
            torch.zeros_like(pred_alpha),
        )
        mask_loss = error_alpha.sum() / pred_mask.sum()

        results_dict.update(
            {
                "gt_alpha": gt_alpha,
                "error_alpha": error_alpha,
            }
        )

        # --------
        photo_loss = color_loss + mask_loss
        # photo_loss = color_loss
        # photo_loss = mask_loss
        return photo_loss, results_dict
    
    def compute_photometric_energy_direct(
        self,
        sample,
        verts,
        faces,
        albedos,
        rast_dict,
        step_i=None,
        stage=None,
        include_keyframes=False,
    ):
        """
        Computes the dense photometric energy
        :param sample:
        :param vertices:
        :param albedos:
        :return:
        """
        gt_rgb = sample["rgb"].to(verts)
        if "alpha" in sample:
            gt_alpha = sample["alpha_map"].to(verts)
        else:
            gt_alpha = None

        lights = self.lights[None] if self.lights is not None else None
        bg_color = self.get_background_color(gt_rgb, gt_alpha, stage)

        align_texture_except_fid = self.flame.mask.get_fid_by_region(
            self.cfg.stages[stage].align_texture_except
        ) if stage is not None else None
        align_boundary_except_vid = self.flame.mask.get_vid_by_region(
            self.cfg.stages[stage].align_boundary_except
        ) if stage is not None else None

        render_out = self.render_rgba(
            rast_dict, verts, faces, albedos, lights, bg_color, 
            align_texture_except_fid, align_boundary_except_vid,
        )

        pred_rgb = render_out['rgba'][:, :3]
        pred_alpha = render_out['rgba'][:, 3:]
        pred_mask = render_out['rgba'][:, [3]].detach() > 0
        pred_mask = pred_mask.expand(-1, 3, -1, -1)

        results_dict = render_out

        # ---- rgb loss ----
        error_rgb = gt_rgb - pred_rgb
        color_loss = error_rgb.abs().sum() / pred_mask.detach().sum()

        results_dict.update(
            {
                "gt_rgb": gt_rgb,
                "pred_rgb": pred_rgb,
                "error_rgb": error_rgb,
                "pred_alpha": pred_alpha,
            }
        )

        # --------
        photo_loss = color_loss
        return photo_loss, results_dict
    
    def compute_regularization_energy(self, result_dict, verts, verts_cano, lmks, albedos, frame_idx, include_keyframes, stage):
        """
        Computes the energy term that penalizes strong deviations from the flame base model
        """
        log_dict = {}
        
        std_tex = 1
        std_expr = 1
        std_shape = 1

        indices = self.select_frame_indices(frame_idx, include_keyframes)

        # pose smoothness term
        if self.opt_dict['pose'] and 'tracking' in stage:
            E_pose_smooth = self.compute_pose_smooth_energy(frame_idx, stage=='global_tracking')
            log_dict["pose_smooth"] = E_pose_smooth

        # joint regularization term
        if self.opt_dict['joints']:
            if 'tracking' in stage:
                joint_smooth = self.compute_joint_smooth_energy(frame_idx, stage=='global_tracking')
                log_dict["joint_smooth"] = joint_smooth

            joint_prior = self.compute_joint_prior_energy(frame_idx)
            log_dict["joint_prior"] = joint_prior

        # expression regularization
        if self.opt_dict['expr']:
            expr = to_batch(self.expr, indices)
            reg_expr = (expr / std_expr) ** 2
            log_dict["reg_expr"] = self.cfg.w.reg_expr * reg_expr.mean()

        # shape regularization
        if self.opt_dict['shape']:
            reg_shape = (self.shape / std_shape) ** 2
            log_dict["reg_shape"] = self.cfg.w.reg_shape * reg_shape.mean()

        # texture regularization
        if self.opt_dict['texture']:
            # texture space
            if not self.cfg.model.tex_painted:
                reg_tex_pca = (self.tex_pca / std_tex) ** 2
                log_dict["reg_tex_pca"] = self.cfg.w.reg_tex_pca * reg_tex_pca.mean()

            # texture map
            if self.cfg.model.tex_extra:
                if self.cfg.w.reg_tex_var is not None:
                    if self.cfg.stages[stage]['reg_tex_var'] is not None:
                        w_reg_tex_var = self.cfg.stages[stage]['reg_tex_var']
                    else:
                        w_reg_tex_var = self.cfg.w.reg_tex_var

                    masks = []

                    for region in self.cfg.model.tex_clusters:
                        mask = self.flame_uvmask.get_uvmask_by_region([region])
                        masks.append(mask[..., None])

                    masks = torch.cat(masks, dim=-1)[None, ...]  # (1, H, W, num_regions)
                    tex_regions = self.get_albedo()[0, ..., None] * masks  # (3, H, W, num_regions)
                    sum_regions = tex_regions.sum(dim=(1, 2), keepdim=True)  # (3, 1, 1, num_regions)
                    mean_regions = sum_regions / masks.sum(dim=(1, 2), keepdim=True)  # (3, 1, 1, num_regions)
                    var_regions = ((tex_regions - mean_regions) ** 2) * masks
                    log_dict["reg_tex_var"] = w_reg_tex_var * var_regions.mean()
                
                if self.cfg.model.residual_tex:
                    if self.cfg.w.reg_tex_res is not None:
                        reg_tex_res = self.tex_extra ** 2
                        # reg_tex_res = self.tex_extra.abs()  # L1 loss can create noise textures

                        # if len(self.cfg.model.occluded) > 0:
                        #     mask = (~self.flame_uvmask.get_uvmask_by_region(self.cfg.model.occluded)).float()[None, ...]
                        #     reg_tex_res *= mask
                        log_dict["reg_tex_res"] = self.cfg.w.reg_tex_res * reg_tex_res.mean()
                    
                    if self.cfg.w.reg_tex_tv is not None:
                        tex = self.get_albedo()[0]  # (3, H, W)
                        tv_y = (tex[..., :-1, :] - tex[..., 1:, :]) ** 2
                        tv_x = (tex[..., :, :-1] - tex[..., :, 1:]) ** 2
                        tv = tv_y.reshape(tv_y.shape[0], -1) + tv_x.reshape(tv_x.shape[0], -1)
                        w_reg_tex_tv = self.cfg.w.reg_tex_tv * self.cfg.data.scale_factor ** 2
                        if self.cfg.data.n_downsample_rgb is not None:
                            w_reg_tex_tv /= (self.cfg.data.n_downsample_rgb ** 2)
                        log_dict["reg_tex_tv"] = w_reg_tex_tv * tv.mean()
                    
                    if self.cfg.w.reg_tex_res_clusters is not None:
                        mask_sclerae = self.flame_uvmask.get_uvmask_by_region(self.cfg.w.reg_tex_res_for)[None, :, :]
                        reg_tex_res_clusters = self.tex_extra ** 2 * mask_sclerae
                        log_dict["reg_tex_res_clusters"] = self.cfg.w.reg_tex_res_clusters * reg_tex_res_clusters.mean()

        # lighting parameters regularization
        if self.opt_dict['lights']:
            if self.cfg.w.reg_light is not None and self.lights is not None:
                reg_light = (self.lights - self.lights_uniform) ** 2
                log_dict["reg_light"] = self.cfg.w.reg_light * reg_light.mean()
            
            if self.cfg.w.reg_diffuse is not None and self.lights is not None:
                diffuse = result_dict['diffuse_detach_normal']
                reg_diffuse = F.relu(diffuse.max() - 1) + diffuse.var(dim=1).mean()
                log_dict["reg_diffuse"] = self.cfg.w.reg_diffuse * reg_diffuse
            
        # offset regularization
        if self.opt_dict['static_offset'] or self.opt_dict['dynamic_offset']:
            if self.static_offset is not None or self.dynamic_offset is not None:
                offset = 0
                if self.static_offset is not None:
                    offset += self.static_offset
                if self.dynamic_offset is not None:
                    offset += to_batch(self.dynamic_offset, indices)

                if self.cfg.w.reg_offset_lap is not None:
                    # laplacian loss
                    vert_wo_offset = (verts_cano - offset).detach()
                    reg_offset_lap = self.compute_laplacian_smoothing_loss(
                        vert_wo_offset, vert_wo_offset + offset
                    )
                    if len(self.cfg.w.reg_offset_lap_relax_for) > 0:
                        w = self.scale_vertex_weights_by_region(
                            weights=torch.ones_like(verts[:, :, :1]),
                            scale_factor=self.cfg.w.reg_offset_lap_relax_coef,
                            region=self.cfg.w.reg_offset_lap_relax_for,
                        )
                        reg_offset_lap *= w
                    log_dict["reg_offset_lap"] = self.cfg.w.reg_offset_lap * reg_offset_lap.mean()

                if self.cfg.w.reg_offset is not None:
                    # norm loss
                    # reg_offset = offset.norm(dim=-1, keepdim=True)
                    reg_offset = offset.abs()
                    if len(self.cfg.w.reg_offset_relax_for) > 0:
                        w = self.scale_vertex_weights_by_region(
                            weights=torch.ones_like(verts[:, :, :1]),
                            scale_factor=self.cfg.w.reg_offset_relax_coef,
                            region=self.cfg.w.reg_offset_relax_for,
                        )
                        reg_offset *= w
                    log_dict["reg_offset"] = self.cfg.w.reg_offset * reg_offset.mean()
                
                if self.cfg.w.reg_offset_rigid is not None:
                    reg_offset_rigid = 0
                    for region in self.cfg.w.reg_offset_rigid_for:
                        vids = self.flame.mask.get_vid_by_region([region])
                        reg_offset_rigid += offset[:, vids, :].var(dim=-2).mean()
                    log_dict["reg_offset_rigid"] = self.cfg.w.reg_offset_rigid * reg_offset_rigid

                if self.cfg.w.reg_offset_dynamic is not None and self.dynamic_offset is not None and self.opt_dict['dynamic_offset']:
                    # The dynamic offset is regularized to be temporally smooth
                    if frame_idx == 0:
                        reg_offset_d = torch.zeros_like(self.dynamic_offset[0])
                        offset_d = self.dynamic_offset[0]
                    else:
                        reg_offset_d = torch.stack([self.dynamic_offset[0], self.dynamic_offset[frame_idx - 1]])
                        offset_d = self.dynamic_offset[frame_idx]

                    reg_offset_dynamic = ((offset_d - reg_offset_d) ** 2).mean()
                    log_dict["reg_offset_dynamic"] = self.cfg.w.reg_offset_dynamic * reg_offset_dynamic

                if self.cfg.w.reg_offset_eyes is not None:
                    # eye offset loss to prevent eyeballs from penetrating the eyesockets
                    reg_offset_eyes = self.compute_eye_region_offset_energy(verts_cano)
                    log_dict["reg_offset_eyes"] = self.cfg.w.reg_offset_eyes * reg_offset_eyes
                        
        return log_dict

    def scale_vertex_weights_by_region(self, weights, scale_factor, region):
        indices = self.flame.mask.get_vid_by_region(region)
        weights[:, indices] *= scale_factor

        for _ in range(self.cfg.w.blur_iter):
            M = self.flame.laplacian_matrix_negate_diag[None, ...]
            weights = M.bmm(weights) / 2
        return weights
    
    def compute_eye_region_offset_energy(self, verts: torch.Tensor):
        """ Regularize the offset of the eye region in the canonical space 
            so that the eyeballs are not penetrating the eyesockets.

            Args:
                verts: (B, V, 3) tensor of vertices in the canonical space
        """
        # prevent eyeballs from penetrating the eyesockets (left eye)
        vid_eyeball_l = self.flame.mask.get_vid_by_region(["left_eyeball"])
        left_eyeball_center = verts[:, vid_eyeball_l, :].mean(dim=1, keepdim=True)

        vid_eye_l = self.flame.mask.get_vid_by_region(["left_eye_region"])
        left_eye_region = verts[:, vid_eye_l, :]
        o = self.static_offset[:, vid_eye_l, :]

        v2c = (left_eyeball_center - left_eye_region)
        v2c = v2c / torch.norm(v2c, dim=-1, keepdim=True)  # direction from a vertex to the center
        d_l = (o * v2c).sum(dim=-1, keepdim=True) * v2c  # projection of offset onto the direction

        # prevent eyeballs from penetrating the eyesockets (right eye)
        vid_eyeball_r = self.flame.mask.get_vid_by_region(["right_eyeball"])
        right_eyeball_center = verts[:, vid_eyeball_r, :].mean(dim=1, keepdim=True)

        vid_eye_r = self.flame.mask.get_vid_by_region(["right_eye_region"])
        right_eye_region = verts[:, vid_eye_r, :]
        o = self.static_offset[:, vid_eye_r, :]

        v2c = (right_eyeball_center - right_eye_region)
        v2c = v2c / torch.norm(v2c, dim=-1, keepdim=True)  # direction from a vertex to the center
        d_r = (o * v2c).sum(dim=-1, keepdim=True) * v2c  # projection of offset onto the direction

        # summation
        return F.relu(d_l).mean() + F.relu(d_r).mean()

    def compute_pose_smooth_energy(self, frame_idx, use_next_frame=False):
        """
        Regularizes the global pose of the flame head model to be temporally smooth
        """
        idx = frame_idx
        idx_prev = np.clip(idx - 1, 0, self.n_timesteps - 1)
        if use_next_frame:
            idx_next = np.clip(idx + 1, 0, self.n_timesteps - 1)
            ref_indices = [idx_prev, idx_next]
        else:
            ref_indices = [idx_prev]

        E_trans = ((self.translation[[idx]] - self.translation[ref_indices].detach()) ** 2).mean() * self.cfg.w.smooth_trans
        E_rot = ((self.rotation[[idx]] - self.rotation[ref_indices].detach()) ** 2).mean() * self.cfg.w.smooth_rot
        return E_trans + E_rot
    
    def compute_joint_smooth_energy(self, frame_idx, use_next_frame=False):
        """
        Regularizes the joints of the flame head model to be temporally smooth
        """
        idx = frame_idx
        idx_prev = np.clip(idx - 1, 0, self.n_timesteps - 1)
        if use_next_frame:
            idx_next = np.clip(idx + 1, 0, self.n_timesteps - 1)
            ref_indices = [idx_prev, idx_next]
        else:
            ref_indices = [idx_prev]

        E_joint_smooth = 0
        E_joint_smooth += ((self.neck_pose[[idx]] - self.neck_pose[ref_indices].detach()) ** 2).mean() * self.cfg.w.smooth_neck
        E_joint_smooth += ((self.jaw_pose[[idx]] - self.jaw_pose[ref_indices].detach()) ** 2).mean() * self.cfg.w.smooth_jaw
        E_joint_smooth += ((self.eyes_pose[[idx]] - self.eyes_pose[ref_indices].detach()) ** 2).mean() * self.cfg.w.smooth_eyes
        if self.flame.has_neck_base_joint:
            E_joint_smooth += ((self.neck_base_pose[[idx]] - self.neck_base_pose[ref_indices].detach()) ** 2).mean() * self.cfg.w.smooth_neck_base
        return E_joint_smooth
    
    def compute_joint_prior_energy(self, frame_idx):
        """
        Regularizes the joints of the flame head model towards neutral joint locations
        """
        poses = [
            ("neck", self.neck_pose[[frame_idx], :]),
            ("jaw", self.jaw_pose[[frame_idx], :]),
            ("eyes", self.eyes_pose[[frame_idx], :3]),
            ("eyes", self.eyes_pose[[frame_idx], 3:]),
        ]
        if self.flame.has_neck_base_joint:
            poses.append(("neck_base", self.neck_base_pose[[[frame_idx]]]))
       
        # Joints should are regularized towards neural
        E_joint_prior = 0
        for name, pose in poses:
            rotmats = batch_rodrigues(torch.cat([torch.zeros_like(pose), pose], dim=0))
            diff = ((rotmats[[0]] - rotmats[1:]) ** 2).mean()

            if name == 'jaw':
                # penalize negative rotation along x axis of jaw for physical plausibility
                diff += F.relu(-pose[:, 0]).mean() * 10
            elif name == 'eyes':
                diff += ((self.eyes_pose[[frame_idx], :3] - self.eyes_pose[[frame_idx], 3:]) ** 2).mean()

            E_joint_prior += diff * self.cfg.w[f"prior_{name}"]
        return E_joint_prior

    def compute_laplacian_smoothing_loss(self, verts, offset_verts):
        L = self.flame.laplacian_matrix[None, ...].detach()  # (1, V, V)
        basis_lap = L.bmm(verts).detach()  #.norm(dim=-1) * weights

        offset_lap = L.bmm(offset_verts)  #.norm(dim=-1) # * weights
        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff

    def compute_energy(
        self,
        sample,
        frame_idx,
        include_keyframes=False,
        step_i=None,
        stage=None,
    ):
        """
        Compute total energy for frame frame_idx
        :param sample:
        :param frame_idx:
        :param include_keyframes: if key frames shall be included when predicting the per
        frame energy
        :return: loss, log dict, predicted vertices and landmarks
        """
        log_dict = {}

        gt_rgb = sample["rgb"]
        result_dict = {"gt_rgb": gt_rgb}

        verts, verts_cano, lmks, albedos = self.forward_flame(frame_idx, include_keyframes)
        faces = self.flame.faces

        if isinstance(sample["num_cameras"], list):
            num_cameras = sample["num_cameras"][0]
        else:
            num_cameras = sample["num_cameras"]
        # albedos = self.repeat_n_times(albedos, num_cameras)  # only needed for pytorch3d renderer

        if self.cfg.w.landmark is not None:
            lmks_n = self.repeat_n_times(lmks, num_cameras)
            E_lmk, _result_dict = self.compute_lmk_energy(sample, lmks_n)
            log_dict["lmk"] = self.cfg.w.landmark * E_lmk
            result_dict.update(_result_dict)
        
        if stage is None or 'rgb' in stage:
            if self.cfg.w.photo is not None:
                verts_n = self.repeat_n_times(verts, num_cameras)
                rast_dict = self.rasterize_flame(
                    sample, verts_n, self.flame.faces, train_mode=True
                )

                if self.cfg.render.diff_rast:
                    photo_energy_func = self.compute_photometric_energy_direct
                else:
                    photo_energy_func = self.compute_photometric_energy
                E_photo, _result_dict = photo_energy_func(
                    sample,
                    verts,
                    faces, 
                    albedos,
                    rast_dict,
                    step_i,
                    stage,
                    include_keyframes,
                )
                result_dict.update(_result_dict)
                log_dict["photo"] = self.cfg.w.photo * E_photo
        
        if stage is not None:
            _log_dict = self.compute_regularization_energy(
                result_dict, verts, verts_cano, lmks, albedos, frame_idx, include_keyframes, stage
            )
            log_dict.update(_log_dict)

        E_total = torch.stack([v for k, v in log_dict.items()]).sum()
        log_dict["total"] = E_total

        return E_total, log_dict, verts, faces, lmks, albedos, result_dict

    def repeat_n_times(self, x: torch.Tensor, n: int):
        """Expand a tensor from shape [F, ...] to [F*n, ...]"""
        return x.unsqueeze(1).repeat_interleave(n, dim=1).reshape(-1, *x.shape[1:])

    @torch.no_grad()
    def log_scalars(
        self, 
        log_dict, 
        frame_idx, 
        session: Literal["train", "eval"] = "train", 
        stage=None,
        frame_step=None, 
        # step_in_stage=None, 
    ):
        """
        Logs scalars in log_dict to tensorboard and logger
        :param log_dict:
        :param frame_idx:
        :param step_i:
        :return:
        """

        log_msg = ""

        if session == "train":
            # assert step_in_stage is not None
            # for k, v in self.decays.items():
            #     decay = v.get(step_in_stage)
            #     log_dict[f"decay_{k}"] = decay
            global_step = self.global_step
        else:
            global_step = frame_idx

        for k, v in log_dict.items():
            if not k.startswith("decay"):
                log_msg += "{}: {:.4f}  ".format(k, v)
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(f"{session}/{k}", v, global_step)

        if session == "train":
            assert stage is not None
            if frame_step is not None:
                msg_prefix = f"[{session}-{stage}] frame {frame_idx} step {frame_step}:  "
            else:
                msg_prefix = f"[{session}-{stage}] frame {frame_idx} step {self.global_step}:  "
        elif session == "eval":
            msg_prefix = f"[{session}] frame {frame_idx}:  "
        logger.info(msg_prefix + log_msg)

    def save_obj_with_texture(self, vertices, faces, uv_coordinates, uv_indices, albedos, obj_path, mtl_path, texture_path):
        # Save the texture image
        torchvision.utils.save_image(albedos.squeeze(0), texture_path)

        # Create the MTL file
        with open(mtl_path, 'w') as f:
            f.write(get_mtl_content(texture_path.name))
        
        # Create the obj file
        with open(obj_path, 'w') as f:
            f.write(get_obj_content(vertices, faces, uv_coordinates, uv_indices, mtl_path.name))
    
    def async_func(func):
        """Decorator to run a function asynchronously"""
        def wrapper(*args, **kwargs):
            self = args[0]
            if self.cfg.async_func:
                thread = threading.Thread(target=func, args=args, kwargs=kwargs)
                thread.start()
            else:
                func(*args, **kwargs)
        return wrapper
    
    @torch.no_grad()
    @async_func
    def log_media(
        self,
        verts: torch.tensor,
        faces: torch.tensor,
        lmks: torch.tensor,
        albedos: torch.tensor,
        output_dict: dict,
        sample: dict,
        frame_idx: int,
        session: str,
        stage: Optional[str]=None,
        frame_step: int=None,
        epoch=None,
    ):
        """
        Logs current tracking visualization to tensorboard
        :param verts:
        :param lmks:
        :param sample:
        :param frame_idx:
        :param frame_step:
        :param show_lmks:
        :param show_overlay:
        :return:
        """
        tic = time.time()
        prepare_output_path = partial(
            self.prepare_output_path, 
            session=session, 
            frame_idx=frame_idx, 
            stage=stage, 
            step=frame_step,
            epoch=epoch,
        )

        """images"""
        img = self.visualize_tracking(verts, lmks, albedos, output_dict, sample)
        img_path = prepare_output_path(folder_name="image_grid", file_type=self.cfg.log.image_format)
        torchvision.utils.save_image(img, img_path)

        """meshes"""
        texture_path = prepare_output_path(folder_name="mesh", file_type=self.cfg.log.image_format)
        mtl_path = prepare_output_path(folder_name="mesh", file_type="mtl")
        obj_path = prepare_output_path(folder_name="mesh", file_type="obj")
    
        vertices = verts.squeeze(0).detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        uv_coordinates = self.flame.verts_uvs.cpu().numpy()
        uv_indices = self.flame.textures_idx.cpu().numpy()
        self.save_obj_with_texture(vertices, faces, uv_coordinates, uv_indices, albedos, obj_path, mtl_path, texture_path)
        """"""
    
        # log_figure = self.visualize_flame_multiview(verts, faces, albedos, sample)
        # self.save_image(
        #     log_figure,
        #     frame_idx,
        #     folder_name=f"{session}/flame_multiview",
        #     step=frame_step,
        # )

        # log_figure = self.visualize_trajectory(sample)
        # self.tb_writer.add_image("translation_trajectory", log_figure, frame_stepd)
        # self.save_image(
        #     log_figure, frame_idx, folder_name=f"{session}/translation_trajectory", step=frame_step
        # )

        toc = time.time() - tic
        if stage is not None:
            msg_prefix = f"[{session}-{stage}] frame {frame_idx}"
        else:
            msg_prefix = f"[{session}] frame {frame_idx}"
        if frame_step is not None:
            msg_prefix += f" step {frame_step}"
        logger.info(f"{msg_prefix}:  Logging media took {toc:.2f}s")

    @torch.no_grad()
    def visualize_flame_multiview(self, verts, faces, albedos, sample):

        # prepare sample to be used with three instances
        new_sample = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                new_sample[k] = v[[0, 0, 0]]
            else:
                new_sample[k] = v
        sample = new_sample

        # rotate vertices to view them from center, left and right
        frame_idx = sample["timestep_index"][0]
        verts = verts[0]

        # subtract translation before rotating
        t = self.translation[frame_idx][None]
        verts_center = verts - t

        # rotate
        turn_90 = batch_rodrigues(
            torch.tensor([[0, np.pi / 2, 0]], device=self.device).float()
        )[0]
        verts_right = (turn_90 @ verts_center.T).T
        verts_left = (turn_90.T @ verts_center.T).T

        # add translation back to the vertices
        verts = (
            torch.stack([verts_center, verts_right, verts_left]) + t[None]
        )

        # render
        rast_dict = self.rasterize_flame(
            sample, verts, faces, camera_index=[0, 0, 0], train_mode=False, 
        )

        lights = self.lights[None] if self.lights is not None else None

        # render RGB
        render_out = self.render_rgba(
            rast_dict, verts, faces, albedos[[0, 0, 0]], lights
        )
        rgba = render_out["rgba"]
        normal = render_out["normal"]

        predicted_images = torch.cat([rgba[..., :3], normal], dim=0)
        return torchvision.utils.make_grid(predicted_images, nrow=3)

    @torch.no_grad()
    def visualize_tracking(
        self,
        verts,
        lmks,
        albedos,
        output_dict,
        sample,
        return_imgs_seperately=False,
    ):
        """
        Visualizes the tracking result
        """
        if len(self.cfg.log.view_indices) > 0:
            view_indices = torch.tensor(self.cfg.log.view_indices)
        else:
            num_views = sample["rgb"].shape[0]
            if num_views > 1:
                step = (num_views - 1) // (self.cfg.log.max_num_views - 1)
                view_indices = torch.arange(0, num_views, step=step)
            else:
                view_indices = torch.tensor([0])
        num_views_log = len(view_indices)

        imgs = []

        # rgb
        gt_rgb = output_dict["gt_rgb"][view_indices].cpu()
        transfm = torchvision.transforms.Resize(gt_rgb.shape[-2:])
        imgs += [img[None] for img in gt_rgb]

        if "pred_rgb" in output_dict:
            pred_rgb = transfm(output_dict["pred_rgb"][view_indices].cpu())
            pred_rgb = torch.clip(pred_rgb, min=0, max=1)
            imgs += [img[None] for img in pred_rgb]

        if "error_rgb" in output_dict:
            error_rgb = transfm(output_dict["error_rgb"][view_indices].cpu())
            error_rgb = error_rgb.mean(dim=1) / 2 + 0.5
            cmap = cm.get_cmap("seismic")
            error_rgb = cmap(error_rgb.cpu())
            error_rgb = torch.from_numpy(error_rgb[..., :3]).to(gt_rgb).permute(0, 3, 1, 2)
            imgs += [img[None] for img in error_rgb]
        
        # cluster id
        if "cid" in output_dict:
            cid = transfm(output_dict["cid"][view_indices].cpu())
            cid = cid / cid.max()
            cid = cid.expand(-1, 3, -1, -1).clone()

            pred_alpha = transfm(output_dict["pred_alpha"][view_indices].cpu()).expand(-1, 3, -1, -1)
            bg = pred_alpha == 0
            cid[bg] = 1
            imgs += [img[None] for img in cid]
        
        # albedo
        if "albedo" in output_dict:
            albedo = transfm(output_dict["albedo"][view_indices].cpu())
            albedo = torch.clip(albedo, min=0, max=1)

            pred_alpha = transfm(output_dict["pred_alpha"][view_indices].cpu()).expand(-1, 3, -1, -1)
            bg = pred_alpha == 0
            albedo[bg] = 1
            imgs += [img[None] for img in albedo]
        
        # normal
        if "normal" in output_dict:
            normal = transfm(output_dict["normal"][view_indices].cpu())
            normal = torch.clip(normal/2+0.5, min=0, max=1)
            imgs += [img[None] for img in normal]
        
        # diffuse
        diffuse = None
        if self.cfg.render.lighting_type != 'constant' and "diffuse" in output_dict:
            diffuse = transfm(output_dict["diffuse"][view_indices].cpu())
            diffuse = torch.clip(diffuse, min=0, max=1)
            imgs += [img[None] for img in diffuse]
        
        # aa
        if "aa" in output_dict:
            aa = transfm(output_dict["aa"][view_indices].cpu())
            aa = torch.clip(aa, min=0, max=1)
            imgs += [img[None] for img in aa]

        # alpha
        if "gt_alpha" in output_dict:
            gt_alpha = transfm(output_dict["gt_alpha"][view_indices].cpu()).expand(-1, 3, -1, -1)
            imgs += [img[None] for img in gt_alpha]

        if "pred_alpha" in output_dict:
            pred_alpha = transfm(output_dict["pred_alpha"][view_indices].cpu()).expand(-1, 3, -1, -1)
            color_alpha = torch.tensor([0.2, 0.5, 1])[None, :, None, None]
            fg_mask = (pred_alpha > 0).float()
            if diffuse is not None:
                fg_mask *= diffuse
                w = 0.7
            overlay_alpha = fg_mask * (w * color_alpha * pred_alpha + (1-w) * gt_rgb) \
                + (1 - fg_mask) * gt_rgb
            imgs += [img[None] for img in overlay_alpha]

        if "error_alpha" in output_dict:
            error_alpha = transfm(output_dict["error_alpha"][view_indices].cpu())
            error_alpha = error_alpha.mean(dim=1) / 2 + 0.5
            cmap = cm.get_cmap("seismic")
            error_alpha = cmap(error_alpha.cpu())
            error_alpha = (
                torch.from_numpy(error_alpha[..., :3]).to(gt_rgb).permute(0, 3, 1, 2)
            )
            imgs += [img[None] for img in error_alpha]
        else:
            error_alpha = None
        
        # landmark
        vis_lmk = self.visualize_landmarks(gt_rgb, output_dict, view_indices)
        if vis_lmk is not None:
            imgs += [img[None] for img in vis_lmk]
        # ----------------
        num_types = len(imgs) // len(view_indices)
        
        if return_imgs_seperately:
            return imgs
        else:
            if self.cfg.log.stack_views_in_rows:
                imgs = [imgs[j * num_views_log + i] for i in range(num_views_log) for j in range(num_types)]
                imgs = torch.cat(imgs, dim=0).cpu()
                return torchvision.utils.make_grid(imgs, nrow=num_types)
            else:
                imgs = torch.cat(imgs, dim=0).cpu()
                return torchvision.utils.make_grid(imgs, nrow=num_views_log)
    
    @torch.no_grad()
    def visualize_mesh(self, img, output_dict, view_indices=torch.tensor([0])):
        # normal
        if "normal" in output_dict:
            normal = output_dict["normal"][view_indices]
            normal = torch.clip(normal/2+0.5, min=0, max=1)
            img = normal

        # alpha
        if "pred_alpha" in output_dict:
            pred_alpha = output_dict["pred_alpha"][view_indices].expand(-1, 3, -1, -1)

            img = (normal * pred_alpha + (1 - pred_alpha) * img.cuda()).cpu()

            # color_alpha = torch.tensor([0.2, 0.5, 1])[None, :, None, None]
            # return color_alpha * color_alpha * pred_alpha + (1 - pred_alpha) * gt_rgb

        return img

    @torch.no_grad()
    def visualize_landmarks(self, gt_rgb, output_dict, view_indices=torch.tensor([0])):
        h, w = gt_rgb.shape[-2:]
        unit = h / 750
        wh = torch.tensor([[[w, h]]])
        vis_lmk = None
        if "gt_lmk2d" in output_dict:
            gt_lmk2d = (output_dict['gt_lmk2d'][view_indices].cpu() * 0.5 + 0.5) * wh
            if self.cfg.w.disable_boundary_landmarks:
                gt_lmk2d = gt_lmk2d[:, 17:68]
            else:
                gt_lmk2d = gt_lmk2d[:, :68]
            vis_lmk = gt_rgb.clone() if vis_lmk is None else vis_lmk
            for i in range(len(view_indices)):
                vis_lmk[i] = plot_landmarks_2d(
                    vis_lmk[i].clone(),
                    gt_lmk2d[[i]], 
                    colors="green",
                    unit=unit,
                    input_float=True, 
                ).to(vis_lmk[i])
        if "pred_lmk2d" in output_dict:
            pred_lmk2d = (output_dict['pred_lmk2d'][view_indices].cpu() * 0.5 + 0.5) * wh
            if self.cfg.w.disable_boundary_landmarks:
                pred_lmk2d = pred_lmk2d[:, 17:68]
            else:
                pred_lmk2d = pred_lmk2d[:, :68]
            vis_lmk = gt_rgb.clone() if vis_lmk is None else vis_lmk
            for i in range(len(view_indices)):
                vis_lmk[i] = plot_landmarks_2d(
                    vis_lmk[i].clone(),
                    pred_lmk2d[[i]], 
                    colors="red",
                    unit=unit,
                    input_float=True, 
                ).to(vis_lmk[i])
        return vis_lmk
        

    @torch.no_grad()
    def visualize_trajectory(self, sample):
        """
        Visualizes the trajectory of the tracked model in camera space. That is the trajectory
        of the translation vectors
        :param sample:
        :return:
        """
        indices = list(range(max(sample["timestep_index"])))
        translations = to_batch(self.translation, indices).detach().cpu().numpy()

        fig = plt.figure(figsize=(2, 2), dpi=100, tight_layout=True)
        ax = fig.add_subplot(projection="3d")

        cmap = cm.get_cmap("Spectral")
        for i in range(len(translations) - 1):
            neighbors = translations[[i, i + 1]]
            ax.plot(
                neighbors[:, 0],
                -neighbors[:, 2],
                neighbors[:, 1],
                color=cmap(i / len(translations)),
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        fig.canvas.draw()

        # convert figure to image
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose(
            2, 0, 1
        )
        return torch.tensor(data, device=self.device, dtype=float) * 2 - 1

    @torch.no_grad()
    def evaluate(self, make_visualization=True, epoch=0):
        # always save parameters before evaluation
        self.export_result(epoch=epoch)

        logger.info("Started Evaluation")
        # vid_frames = []
        photo_loss = []
        for frame_idx in range(self.n_timesteps):

            sample = self.get_current_frame(frame_idx, include_keyframes=False)
            self.clear_cache()
            self.fill_cam_params_into_sample(sample)
            (
                E_total,
                log_dict,
                verts,
                faces,
                lmks,
                albedos,
                output_dict,
            ) = self.compute_energy(sample, frame_idx)

            self.log_scalars(log_dict, frame_idx, session="eval")
            photo_loss.append(log_dict["photo"].item())

            if make_visualization:
                self.log_media(
                    verts,
                    faces,
                    lmks,
                    albedos,
                    output_dict,
                    sample,
                    frame_idx,
                    session="eval",
                    epoch=epoch,
                )
        
        self.tb_writer.add_scalar(f"eval_mean/photo", np.mean(photo_loss), epoch)

        # if make_visualization:
        #     msg_prefix = f"[eval] frame {frame_idx}:  "
        #     logger.info(msg_prefix + "Saving as video")

        #     vid_frames = torch.cat(vid_frames, dim=0)
        #     self.save_video(vid_frames, name="eval/tracking_result")

    def prepare_output_path(self, session, frame_idx, folder_name, file_type, stage=None, step=None, epoch=None):
        if epoch is not None:
            output_folder = self.out_dir / f'{session}_{epoch}' / folder_name
        else:
            output_folder = self.out_dir / session / folder_name
        os.makedirs(output_folder, exist_ok=True)
        
        if stage is not None:
            assert step is not None
            fname = "frame_{:05d}_{:03d}_{}.{}".format(frame_idx, step, stage, file_type)
        else:
            fname = "frame_{:05d}.{}".format(frame_idx, file_type)
        return output_folder / fname

    # def save_video(self, vid_frames, name):
    #     """Save image frames as a video.

    #     Args:
    #         vid_frame:
    #             type: torch.Tensor
    #             shape: (num_frames, height, width, 3)
    #             dtype: torch.uint8
    #         name: str
    #     """
    #     vid_path = str(self.out_dir / f"{name}.mp4")
    #     torchvision.io.write_video(
    #         vid_path,
    #         vid_frames,
    #         fps=self.config["frame_rate"],
    #         options={"crf": "10"},
    #     )


    def export_result(self, fname=None, epoch=None):
        """
        Saves tracked/optimized flame parameters.
        :return:
        """
        # save parameters
        keys = [
            "rotation",
            "translation",
            "neck_pose",
            "jaw_pose",
            "eyes_pose",
            "shape",
            "expr",
            "timestep_id",
            "n_processed_frames",
        ]
        values = [
            self.rotation,
            self.translation,
            self.neck_pose,
            self.jaw_pose,
            self.eyes_pose,
            self.shape,
            self.expr,
            np.array(self.dataset.timestep_ids),
            self.frame_idx,
        ]
        if self.flame.has_neck_base_joint:
            keys += ["neck_base_pose"]
            values += [self.neck_base_pose]

        if not self.calibrated:
            keys += ["K", "RT"]
            values += [self.K, self.RT]
        
        if not self.cfg.model.tex_painted:
            keys += ["tex"]
            values += [self.tex_pca]
        
        if self.cfg.model.tex_extra:
            keys += ["tex_extra"]
            values += [self.tex_extra]
        
        if self.lights is not None:
            keys += ["lights"]
            values += [self.lights]
        
        if self.cfg.model.use_static_offset:
            keys += ["static_offset"]
            values += [self.static_offset]

        if self.cfg.model.use_dynamic_offset:
            keys += ["dynamic_offset"]
            values += [self.dynamic_offset]

        export_dict = {}
        for k, v in zip(keys, values):
            if not isinstance(v, np.ndarray):
                if isinstance(v, list):
                    v = torch.stack(v)
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
            export_dict[k] = v

        export_dict["image_size"] = np.array(self.image_size)

        fname = fname if fname is not None else "tracked_flame_params"
        if epoch is not None:
            fname = f"{fname}_{epoch}"
        np.savez(self.out_dir / f'{fname}.npz', **export_dict)


class GlobalTracker(FlameTracker):
    def __init__(self, cfg: BaseTrackingConfig):
        super().__init__(cfg)

        self.calibrated = cfg.data.calibrated

        self.detect_landmarks(cfg)

        # logging
        out_dir = cfg.exp.output_folder / cfg.exp.name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir.mkdir(parents=True)

        self.frame_idx = self.cfg.begin_frame_idx
        self.out_dir = out_dir
        self.tb_writer = SummaryWriter(self.out_dir)
        
        self.log_interval_scalar = self.cfg.log.interval_scalar
        self.log_interval_media = self.cfg.log.interval_media

        config_yaml_path = out_dir / 'config.yml'
        config_yaml_path.write_text(yaml.dump(cfg), "utf8")
        print(tyro.to_yaml(cfg))

        # data
        self.dataset = MultiViewHeadDataset(
            root_folder=cfg.data.root_folder,
            subject=cfg.data.subject,
            sequence=cfg.data.sequence,
            division=cfg.data.division,
            subset=cfg.data.subset,
            n_downsample_rgb=cfg.data.n_downsample_rgb,
            scale_factor=cfg.data.scale_factor,
            align_cameras_to_axes=cfg.data.align_cameras_to_axes,
            camera_coord_conversion=cfg.data.camera_coord_conversion,
            background_color=cfg.data.background_color,
            use_color_correction=cfg.data.use_color_correction,
            use_alpha_map=cfg.data.use_alpha_map,
            use_landmark=cfg.data.use_landmark,
            landmark_source=cfg.data.landmark_source,
            img_to_tensor=True,
            batchify_all_views=True,  # important to optimized all views together
        )
        # FlameTracker expects all views of a frame in a batch, which is undertaken by the
        # dataset. Therefore batching is disabled for the dataloader

        self.image_size = self.dataset[0]["rgb"].shape[-2:]
        self.n_timesteps = len(self.dataset)

        # parameters
        self.init_params()

        if self.cfg.model.flame_params_path is not None:
            self.load_from_tracked_flame_params(self.cfg.model.flame_params_path)

    def detect_landmarks(self, cfg):
        dataset = MultiViewHeadDataset(
            root_folder=cfg.data.root_folder,
            subject=cfg.data.subject,
            sequence=cfg.data.sequence,
            division=cfg.data.division,
            n_downsample_rgb=cfg.data.n_downsample_rgb,
            scale_factor=cfg.data.scale_factor,
        )

        if cfg.data.landmark_source == 'face-alignment':
            from mvht.util.landmark_detector import LandmarkDetector

            if not cfg.exp.reuse_landmarks or not dataset.get_property_path("face-alignment", 0).exists():
                # LandmarkDetector only supports a batch_size of 1
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

                os.umask(0o002)
                detector = LandmarkDetector()
                detector.annotate_landmarks(dataloader, add_iris=False)
        elif cfg.data.landmark_source == 'star':
            from mvht.util.landmark_detector_star import LandmarkDetectorSTAR
            
            if not cfg.exp.reuse_landmarks or not dataset.get_property_path("landmarks2D/STAR", 0).exists():
                # LandmarkDetector only supports a batch_size of 1
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

                os.umask(0o002)
                detector = LandmarkDetectorSTAR()
                detector.annotate_landmarks(dataloader)
        else:
            raise NotImplementedError(f"Unknown landmark source: {cfg.data.landmark_source}")
    
    def init_params(self):
        train_tensors = []

        # flame model params
        self.shape = torch.zeros(self.cfg.model.n_shape).to(self.device)
        self.expr = torch.zeros(self.n_timesteps, self.cfg.model.n_expr).to(self.device)

        # joint axis angles
        self.neck_pose = torch.zeros(self.n_timesteps, 3).to(self.device)
        self.jaw_pose = torch.zeros(self.n_timesteps, 3).to(self.device)
        self.eyes_pose = torch.zeros(self.n_timesteps, 6).to(self.device)
        if self.flame.has_neck_base_joint:
            self.neck_base_pose = torch.zeros(self.n_timesteps, 3).to(self.device)

        # rigid pose
        self.translation = torch.zeros(self.n_timesteps, 3).to(self.device)
        self.rotation = torch.zeros(self.n_timesteps, 3).to(self.device)

        # texture and lighting params
        self.tex_pca = torch.zeros(self.cfg.model.n_tex).to(self.device)
        if self.cfg.model.tex_extra:
            res = self.cfg.model.tex_resolution
            self.tex_extra = torch.zeros(3, res, res).to(self.device)
        
        if self.cfg.render.lighting_type == 'SH':
            self.lights_uniform = torch.zeros(9, 3).to(self.device)
            self.lights_uniform[0] = torch.tensor([np.sqrt(4 * np.pi)]).expand(3).float().to(self.device)
            self.lights = self.lights_uniform.clone()
        else:
            self.lights = None

        train_tensors += (
            [self.shape, self.translation, self.rotation, self.neck_pose, self.jaw_pose, self.eyes_pose, self.expr,]
        )
        if self.flame.has_neck_base_joint:
            train_tensors += [self.neck_base_pose]

        if not self.cfg.model.tex_painted:
            train_tensors += [self.tex_pca]
        if self.cfg.model.tex_extra:
            train_tensors += [self.tex_extra]

        if self.lights is not None:
            train_tensors += [self.lights]

        if self.cfg.model.use_static_offset:
            self.static_offset = torch.zeros(1, self.flame.v_template.shape[0], 3).to(self.device)
            train_tensors += [self.static_offset]
        else:
            self.static_offset = None
        
        if self.cfg.model.use_dynamic_offset:
            self.dynamic_offset = torch.zeros(self.n_timesteps, self.flame.v_template.shape[0], 3).to(self.device)
            train_tensors += self.dynamic_offset
        else:
            self.dynamic_offset = None

        # camera def inition
        if not self.calibrated:
            # K contains focal length and principle point
            self.K = torch.tensor([1.5]).to(self.device)
            self.RT = torch.eye(3, 4).to(self.device)
            self.RT[2, 3] = -1  # (0, 0, -1) in w2c corresponds to (0, 0, 1) in c2w
            train_tensors += [self.K]

        for t in train_tensors:
            t.requires_grad = True

    def optimize(self):
        """
        Optimizes flame parameters on all frames of the dataset with random rampling
        :return:
        """
        self.global_step = 0
        
        # first initialize frame either from calibration or previous frame
        # with torch.no_grad():
            # self.initialize_frame(frame_idx)

        # sequential optimization of timesteps
        logger.info(f"Start sequential tracking FLAME in {self.n_timesteps} frames")
        dataloader = DataLoader(self.dataset, batch_size=None, shuffle=False, num_workers=4)
        for sample in dataloader:
            timestep = sample["timestep_index"][0].item()
            if timestep == 0:
                self.optimize_stage('lmk_rigid', sample)
                self.optimize_stage('lmk', sample)
                self.optimize_stage('rgb_texture', sample)
                self.optimize_stage('rgb', sample)
                if self.cfg.model.use_static_offset:
                    self.optimize_stage('rgb_offset', sample)

            self.optimize_stage('rgb_sequential_tracking', sample)
            self.initialize_next_timtestep(timestep)
        
        self.evaluate(make_visualization=True, epoch=0)

        logger.info(f"Start global optimization of all frames")
        # global optimization with random sampling
        dataloader = DataLoader(self.dataset, batch_size=None, shuffle=True, num_workers=8)
        self.optimize_stage(stage='rgb_global_tracking', dataloader=dataloader, lr_scale=0.1)

        # if not self.calibrated:
        #     logger.info(
        #         f"Camera intrinsics optimized after initialization frame: {self.K}"
        #     )
        logger.info("All done.")
    
    def optimize_stage(
            self, 
            stage: Literal['lmk_rigid', 'lmk', 'rgb_texture', 'rgb', 'rgb_offset', 'rgb_sequential_tracking', 'rgb_global_tracking'],
            sample = None,
            dataloader = None,
            lr_scale = 1.0,
        ):
        params = self.get_train_parameters(stage)
        optimizer = self.configure_optimizer(params, lr_scale=lr_scale)

        if sample is not None:
            num_steps = self.cfg.stages[stage].num_steps
            for step_i in range(num_steps):
                self.optimize_iter(sample, optimizer, stage)
        else:
            assert dataloader is not None
            num_epochs = self.cfg.stages[stage].num_epochs
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            for epoch_i in range(num_epochs):
                logger.info(f"EPOCH {epoch_i+1} / {num_epochs}")
                for step_i, sample in enumerate(dataloader):
                    self.optimize_iter(sample, optimizer, stage)
                scheduler.step()

                if (epoch_i + 1) % 10 == 0:
                    self.evaluate(make_visualization=True, epoch=epoch_i+1)
    
    def optimize_iter(self, sample, optimizer, stage):
        # compute loss and update parameters
        self.clear_cache()

        timestep_index = sample["timestep_index"][0]
        self.fill_cam_params_into_sample(sample)
        (
            E_total,
            log_dict,
            verts,
            faces,
            lmks,
            albedos,
            output_dict,
        ) = self.compute_energy(
            sample, frame_idx=timestep_index, stage=stage,
        )
        optimizer.zero_grad()
        E_total.backward()
        optimizer.step()

        # log energy terms and visualize
        if (self.global_step+1) % self.log_interval_scalar == 0:
            self.log_scalars(
                log_dict, 
                timestep_index, 
                session="train", 
                stage=stage, 
                frame_step=self.global_step, 
            )

        if (self.global_step+1) % self.log_interval_media == 0:
            self.log_media(
                verts,
                faces,
                lmks,
                albedos,
                output_dict,
                sample,
                timestep_index, 
                session="train",
                stage=stage,
                frame_step=self.global_step,
            )
        del verts, faces, lmks, albedos, output_dict
        self.global_step += 1


    def get_train_parameters(
        self, stage: Literal['lmk_rigid', 'lmk', 'rgb', 'rgb_offset', 'rgb_sequential_tracking', 'rgb_global_tracking'],
    ):
        """
        Collects the parameters to be optimized for the current frame
        :param frame_idx: frame number
        :param first_frame: if true shape params and camera intrinsics will be optimized as well
        :return: dict of parameters
        """
        opt_dict = {
            "cam": False,
            "pose": False,
            "shape": False,
            "joints": False,
            "expr": False,
            "lights": False,
            "texture": False,
            "static_offset": False,
            "dynamic_offset": False,
        }
        if stage == "lmk_rigid":
            opt_dict["cam"] = True
            opt_dict["pose"] = True
        elif stage == "lmk":
            opt_dict["cam"] = True
            opt_dict["pose"] = True
            opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
        elif stage == "rgb_texture":
            opt_dict["cam"] = True
            opt_dict["shape"] = True
            opt_dict["texture"] = True
            opt_dict["lights"] = True
        elif stage == "rgb":
            opt_dict["cam"] = True
            opt_dict["pose"] = True
            opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
            opt_dict["texture"] = True
            opt_dict["lights"] = True
        elif stage == "rgb_offset":
            opt_dict["cam"] = True
            opt_dict["pose"] = True
            opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
            opt_dict["texture"] = True
            opt_dict["lights"] = True
            opt_dict["static_offset"] = True
        elif stage in ["rgb_sequential_tracking"]:
            # opt_dict["cam"] = True
            opt_dict["pose"] = True
            # opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
            # opt_dict["texture"] = True
            # opt_dict["lights"] = True
            # opt_dict["static_offset"] = True
            opt_dict["dynamic_offset"] = True
        elif stage in ["rgb_global_tracking"]:
            opt_dict["cam"] = True
            opt_dict["pose"] = True
            opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
            opt_dict["texture"] = True
            opt_dict["lights"] = True
            opt_dict["static_offset"] = True
            opt_dict["dynamic_offset"] = True
        else:
            raise NotImplementedError(f"Unknown stage name: {stage}")
        self.opt_dict = opt_dict
        
        """parameter groups"""
        params = {
            "cam": [],
            "translation": [],
            "rotation": [],
            "neck_base": [],
            "neck": [],
            "jaw": [],
            "eyes": [],
            "expr": [],
            "lights": [],
            "static_offset": [],
            "dynamic_offset": [],
        }
            
        # shared properties
        if opt_dict["cam"] and not self.calibrated:
            params["cam"] = [self.K]

        if opt_dict["shape"]:        
            params["shape"] = [self.shape]
        
        if opt_dict["texture"]:        
            if not self.cfg.model.tex_painted:
                params["tex"] = [self.tex_pca]
            if self.cfg.model.tex_extra:
                params["tex_extra"] = [self.tex_extra]

        if opt_dict["static_offset"] and self.cfg.model.use_static_offset:
            params["static_offset"] = [self.static_offset]
        
        if opt_dict["lights"] and self.lights is not None:
            params["lights"] = [self.lights]
            
        # per-frame properties
        if opt_dict["pose"]:
            params["translation"].append(self.translation)
            params["rotation"].append(self.rotation)

        if opt_dict["joints"]:
            params["eyes"].append(self.eyes_pose)
            params["neck"].append(self.neck_pose)
            params["jaw"].append(self.jaw_pose)
            if self.flame.has_neck_base_joint:
                params["neck_base"].append(self.neck_base_pose)

        if opt_dict["expr"]:
            params["expr"].append(self.expr)
        
        if opt_dict["dynamic_offset"] and self.cfg.model.use_dynamic_offset:
            params["dynamic_offset"].append(self.dynamic_offset)

        self.opt_dict = opt_dict
        return params

    def initialize_next_timtestep(self, timestep):
        if timestep < self.n_timesteps - 1:
            self.translation[timestep + 1].data.copy_(self.translation[timestep])
            self.rotation[timestep + 1].data.copy_(self.rotation[timestep])
            self.neck_pose[timestep + 1].data.copy_(self.neck_pose[timestep])
            self.jaw_pose[timestep + 1].data.copy_(self.jaw_pose[timestep])
            self.eyes_pose[timestep + 1].data.copy_(self.eyes_pose[timestep])
            if self.flame.has_neck_base_joint:
                self.neck_base_pose[timestep + 1].data.copy_(self.neck_base_pose[timestep])
            self.expr[timestep + 1].data.copy_(self.expr[timestep])
            if self.cfg.model.use_dynamic_offset:
                self.dynamic_offset[timestep + 1].data.copy_(self.dynamic_offset[timestep])


class OnlineTracker(FlameTracker):
    def __init__(self, cfg: BaseTrackingConfig):
        super().__init__(cfg)
        self.n_timesteps = 1
        self.global_step = 0
        self.calibrated = cfg.data.calibrated

        self.init_params()
        self.init_landmark_detector()
    
    def init_params(self):
        train_tensors = []

        # flame model params
        self.shape = torch.zeros(self.cfg.model.n_shape).to(self.device)
        self.expr = torch.zeros(1, self.cfg.model.n_expr).to(self.device)

        # joint axis angles
        self.neck_pose = torch.zeros(1, 3).to(self.device)
        self.jaw_pose = torch.zeros(1, 3).to(self.device)
        self.eyes_pose = torch.zeros(1, 6).to(self.device)
        if self.flame.has_neck_base_joint:
            self.neck_base_pose = torch.zeros(1, 3).to(self.device)

        # rigid pose
        self.translation = torch.zeros(1, 3).to(self.device)
        self.rotation = torch.zeros(1, 3).to(self.device)

        # texture and lighting params
        self.tex_pca = torch.zeros(self.cfg.model.n_tex).to(self.device)
        if self.cfg.model.tex_extra:
            res = self.cfg.model.tex_resolution
            self.tex_extra = torch.zeros(3, res, res).to(self.device)
        
        if self.cfg.render.lighting_type == 'SH':
            self.lights_uniform = torch.zeros(9, 3).to(self.device)
            self.lights_uniform[0] = torch.tensor([np.sqrt(4 * np.pi)]).expand(3).float().to(self.device)
            self.lights = self.lights_uniform.clone()
        else:
            self.lights = None

        train_tensors += (
            [self.shape, self.translation, self.rotation, self.neck_pose, self.jaw_pose, self.eyes_pose, self.expr,]
        )
        if self.flame.has_neck_base_joint:
            train_tensors += [self.neck_base_pose]

        if not self.cfg.model.tex_painted:
            train_tensors += [self.tex_pca]
        if self.cfg.model.tex_extra:
            train_tensors += [self.tex_extra]

        if self.lights is not None:
            train_tensors += [self.lights]

        if self.cfg.model.use_static_offset:
            self.static_offset = torch.zeros(1, self.flame.v_template.shape[0], 3).to(self.device)
            train_tensors += [self.static_offset]
        else:
            self.static_offset = None
        
        if self.cfg.model.use_dynamic_offset:
            self.dynamic_offset = torch.zeros(1, self.flame.v_template.shape[0], 3).to(self.device)
            train_tensors += self.dynamic_offset
        else:
            self.dynamic_offset = None

        # camera def inition
        if not self.calibrated:
            # K contains focal length and principle point
            self.K = torch.tensor([1.5]).to(self.device)
            self.RT = torch.eye(3, 4).to(self.device)
            self.RT[2, 3] = -1  # (0, 0, -1) in w2c corresponds to (0, 0, 1) in c2w
            train_tensors += [self.K]

        for t in train_tensors:
            t.requires_grad = True
    
    def init_landmark_detector(self):
        if self.cfg.data.landmark_source == 'face-alignment':
            from mvht.util.landmark_detector import LandmarkDetector
            self.detector = LandmarkDetector(face_detector='blazeface')
        elif self.cfg.data.landmark_source == 'star':
            from mvht.util.landmark_detector_star import LandmarkDetectorSTAR
            self.detector = LandmarkDetectorSTAR()
        else:
            raise NotImplementedError(f"Unknown landmark source: {self.cfg.data.landmark_source}")

    def optimize(self, img, visualize=False):
        """
        Optimizes flame parameters on all frames of the dataset with random rampling
        :return:
        """
        
        # first initialize frame either from calibration or previous frame
        # with torch.no_grad():
            # self.initialize_frame(frame_idx)

        bbox, lmk2d = self.detector.detect_single_image(img)

        self.image_size = img.shape[:2]

        item = {
            "timestep_index": torch.tensor([0]).cuda(),
            "num_cameras": torch.tensor([1]).cuda(),
            "rgb": torch.from_numpy(img[None]).cuda().permute(0, 3, 1, 2) / 255,
            "lmk2d": torch.from_numpy(lmk2d[None]).cuda(),
        }
        if "lmk2d" in item:
            item["lmk2d"][..., 0] *= self.image_size[1]
            item["lmk2d"][..., 1] *= self.image_size[0]

        # stage = 'lmk_rigid'
        stage = 'lmk'
        # stage = 'rgb'
        
        params = self.get_train_parameters(stage)
        optimizer = self.configure_optimizer(params)
        img = self.optimize_iter(item, optimizer, stage)
        return (img[0] * 255).byte().permute(1, 2, 0).cpu().numpy()

    def optimize_iter(
            self, 
            sample, 
            optimizer, 
            stage: Literal['lmk_rigid', 'lmk', 'rgb_texture', 'rgb', 'rgb_offset', 'rgb_sequential_tracking', 'rgb_global_tracking'],
):
        # compute loss and update parameters
        self.clear_cache()

        timestep_index = sample["timestep_index"][0]
        self.fill_cam_params_into_sample(sample)
        (
            E_total,
            log_dict,
            verts,
            faces,
            lmks,
            albedos,
            output_dict,
        ) = self.compute_energy(
            sample, frame_idx=timestep_index, stage=stage,
        )
        optimizer.zero_grad()
        E_total.backward()
        optimizer.step()

        img = output_dict["gt_rgb"].cpu()

        # log energy terms and visualize
        if (self.global_step+1) % self.cfg.log.interval_scalar == 0:
            self.log_scalars(
                log_dict, 
                timestep_index, 
                session="train", 
                stage=stage, 
                frame_step=self.global_step, 
            )

        if (self.global_step+1) % self.cfg.log.interval_media == 0:

            img = self.visualize_landmarks(img, output_dict)
            img = self.visualize_mesh(img, output_dict)
            # self.log_media(
            #     verts,
            #     faces,
            #     lmks,
            #     albedos,
            #     output_dict,
            #     sample,
            #     timestep_index, 
            #     session="train",
            #     stage=stage,
            #     frame_step=self.global_step,
            # )
        del verts, faces, lmks, albedos, output_dict

        self.global_step += 1
        return img


    def get_train_parameters(
        self, stage: Literal['lmk_rigid', 'lmk', 'rgb', 'rgb_offset', 'rgb_sequential_tracking', 'rgb_global_tracking'],
    ):
        """
        Collects the parameters to be optimized for the current frame
        :param frame_idx: frame number
        :param first_frame: if true shape params and camera intrinsics will be optimized as well
        :return: dict of parameters
        """
        opt_dict = {
            "cam": False,
            "pose": False,
            "shape": False,
            "joints": False,
            "expr": False,
            "lights": False,
            "texture": False,
            "static_offset": False,
            "dynamic_offset": False,
        }
        if stage == "lmk_rigid":
            opt_dict["cam"] = True
            opt_dict["pose"] = True
        elif stage == "lmk":
            opt_dict["cam"] = True
            opt_dict["pose"] = True
            opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
        elif stage == "rgb_texture":
            opt_dict["cam"] = True
            opt_dict["shape"] = True
            opt_dict["texture"] = True
            opt_dict["lights"] = True
        elif stage == "rgb":
            opt_dict["cam"] = True
            opt_dict["pose"] = True
            opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
            opt_dict["texture"] = True
            opt_dict["lights"] = True
        elif stage == "rgb_offset":
            opt_dict["cam"] = True
            opt_dict["pose"] = True
            opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
            opt_dict["texture"] = True
            opt_dict["lights"] = True
            opt_dict["static_offset"] = True
        elif stage in ["rgb_sequential_tracking"]:
            # opt_dict["cam"] = True
            opt_dict["pose"] = True
            # opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
            # opt_dict["texture"] = True
            # opt_dict["lights"] = True
            # opt_dict["static_offset"] = True
            opt_dict["dynamic_offset"] = True
        elif stage in ["rgb_global_tracking"]:
            opt_dict["cam"] = True
            opt_dict["pose"] = True
            opt_dict["shape"] = True
            opt_dict["joints"] = True
            opt_dict["expr"] = True
            opt_dict["texture"] = True
            opt_dict["lights"] = True
            opt_dict["static_offset"] = True
            opt_dict["dynamic_offset"] = True
        else:
            raise NotImplementedError(f"Unknown stage name: {stage}")
        self.opt_dict = opt_dict
        
        """parameter groups"""
        params = {
            "cam": [],
            "translation": [],
            "rotation": [],
            "neck_base": [],
            "neck": [],
            "jaw": [],
            "eyes": [],
            "expr": [],
            "lights": [],
            "static_offset": [],
            "dynamic_offset": [],
        }
            
        # shared properties
        if opt_dict["cam"] and not self.calibrated:
            params["cam"] = [self.K]

        if opt_dict["shape"]:        
            params["shape"] = [self.shape]
        
        if opt_dict["texture"]:        
            if not self.cfg.model.tex_painted:
                params["tex"] = [self.tex_pca]
            if self.cfg.model.tex_extra:
                params["tex_extra"] = [self.tex_extra]

        if opt_dict["static_offset"] and self.cfg.model.use_static_offset:
            params["static_offset"] = [self.static_offset]
        
        if opt_dict["lights"] and self.lights is not None:
            params["lights"] = [self.lights]
            
        # per-frame properties
        if opt_dict["pose"]:
            params["translation"].append(self.translation)
            params["rotation"].append(self.rotation)

        if opt_dict["joints"]:
            params["eyes"].append(self.eyes_pose)
            params["neck"].append(self.neck_pose)
            params["jaw"].append(self.jaw_pose)
            if self.flame.has_neck_base_joint:
                params["neck_base"].append(self.neck_base_pose)

        if opt_dict["expr"]:
            params["expr"].append(self.expr)
        
        if opt_dict["dynamic_offset"] and self.cfg.model.use_dynamic_offset:
            params["dynamic_offset"].append(self.dynamic_offset)

        self.opt_dict = opt_dict
        return params