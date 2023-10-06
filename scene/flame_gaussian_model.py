import torch
import torch.nn as nn
from vht.model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from pytorch3d.transforms import matrix_to_quaternion


class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, n_shape=300, n_expr=100):
        super().__init__(sh_degree)

        self.flame_model = FlameHead(n_shape, n_expr).cuda()
    
    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        meshes = {**train_meshes, **test_meshes}
        tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
        pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
        
        num_timesteps = max(pose_meshes) + 1
        num_verts = meshes[0]['static_offset'].shape[1]
        self.flame_param = {
            'shape': torch.from_numpy(meshes[0]['shape']),
            'expr': torch.zeros([num_timesteps, meshes[0]['expr'].shape[1]]),
            'rotation': torch.zeros([num_timesteps, 3]),
            'neck_pose': torch.zeros([num_timesteps, 3]),
            'jaw_pose': torch.zeros([num_timesteps, 3]),
            'eyes_pose': torch.zeros([num_timesteps, 6]),
            'translation': torch.zeros([num_timesteps, 3]),
            'static_offset': torch.from_numpy(meshes[0]['static_offset']),
            'dynamic_offset': torch.zeros([num_timesteps, num_verts, 3]),
        }
        self.num_timesteps = num_timesteps

        for i, mesh in pose_meshes.items():
            self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
            self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
            self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
            self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
            self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
            self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
            self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
        
        for k, v in self.flame_param.items():
            self.flame_param[k] = v.float().cuda()
        
        if self.binding is None:
            assert len(self._xyz) == len(self.flame_model.faces)
            self.binding = torch.arange(self._xyz.shape[0]).cuda()
    
    def select_mesh_by_timestep(self, timestep):
        verts = self.flame_model(
            self.flame_param['shape'][None, ...],
            self.flame_param['expr'][[timestep]],
            self.flame_param['rotation'][[timestep]],
            self.flame_param['neck_pose'][[timestep]],
            self.flame_param['jaw_pose'][[timestep]],
            self.flame_param['eyes_pose'][[timestep]],
            self.flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            use_rotation_limits=False,
            return_landmarks=False,
            static_offset=self.flame_param['static_offset'],
            dynamic_offset=self.flame_param['dynamic_offset'][[timestep]],
        )

        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # rotate
        self.face_orien_mat = compute_face_orientation(verts, faces).squeeze(0)
        self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)
    
    # def training_setup(self, training_args):
    #     self.percent_dense = training_args.percent_dense
    #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    #     l = [
    #         {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
    #         {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
    #         {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
    #         {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
    #         {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
    #         {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
    #     ]

    #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    #     self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
    #                                                 lr_final=training_args.position_lr_final*self.spatial_lr_scale,
    #                                                 lr_delay_mult=training_args.position_lr_delay_mult,
    #                                                 max_steps=training_args.position_lr_max_steps)
