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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""  # Path to the source data set
        self._target_path = ""  # Path to the target data set for pose and expression transfer
        self._model_path = ""  # Path to the folder to save trained models
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.bind_to_mesh = False
        self.disable_flame_static_offset = False
        self.not_finetune_flame_params = False
        self.select_camera_id = -1
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # 3D Gaussians
        self.iterations = 600_000  # 30_000 (original)
        self.position_lr_init = 0.005  # (scaled up according to mean triangle scale)  #0.00016 (original)
        self.position_lr_final = 0.00005  # (scaled up according to mean triangle scale) # 0.0000016 (original)
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 600_000  # 30_000 (original)
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.017  # (scaled up according to mean triangle scale)  # 0.005 (original)
        self.rotation_lr = 0.001
        self.densification_interval = 2_000  # 100 (original)
        self.opacity_reset_interval = 60_000 # 3000 (original)
        self.densify_from_iter = 10_000  # 500 (original)
        self.densify_until_iter = 600_000  # 15_000 (original)
        self.densify_grad_threshold = 0.0002
        
        # GaussianAvatars
        self.flame_expr_lr = 1e-3
        self.flame_trans_lr = 1e-6
        self.flame_pose_lr = 1e-5
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_xyz = 1e-2
        self.threshold_xyz = 1.
        self.metric_xyz = False
        self.lambda_scale = 1.
        self.threshold_scale = 0.6
        self.metric_scale = False
        self.lambda_dynamic_offset = 0.
        self.lambda_laplacian = 0.
        self.lambda_dynamic_offset_std = 0  #1.

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
