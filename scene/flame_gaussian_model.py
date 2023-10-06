import os
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
from vht.model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from utils.system_utils import mkdir_p


class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, n_shape=300, n_expr=100):
        super().__init__(sh_degree)

        self.face_center = None  # will be set in select_mesh_by_timestep
        self.binding = None  # will be set in load_meshes

        self.flame_model = FlameHead(n_shape, n_expr).cuda()
    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.binding,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.binding,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        meshes = {**train_meshes, **test_meshes}
        tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
        pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
        
        N = max(pose_meshes) + 1
        self.flame_param = {
            'shape': torch.from_numpy(meshes[0]['shape']),
            'expr': torch.zeros([N, meshes[0]['expr'].shape[1]]),
            'rotation': torch.zeros([N, 3]),
            'neck_pose': torch.zeros([N, 3]),
            'jaw_pose': torch.zeros([N, 3]),
            'eyes_pose': torch.zeros([N, 6]),
            'translation': torch.zeros([N, 3]),
            'static_offset': torch.from_numpy(meshes[0]['static_offset']),
        }
        self.num_timesteps = N

        for i, mesh in pose_meshes.items():
            self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
            self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
            self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
            self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
            self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
            self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
        
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
        )

        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # rotate
        self.face_orien_mat = compute_face_orientation(verts, faces).squeeze(0)
        self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)
    
    @property
    def get_rotation(self):
        # always need to normalize the rotation quaternions before chaining them
        rot = self.rotation_activation(self._rotation)
        face_orien_quat = self.rotation_activation(self.face_orien_quat[self.binding])
        return quaternion_multiply(rot, face_orien_quat)
    
    @property
    def get_xyz(self):
        if self.face_center is None:
            return self._xyz
        else:
            xyz = torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1)
            return xyz + self.face_center[self.binding]
    
    # TODO: do we need to rotate the SH function?
    # @property
    # def get_features(self):
    #     features_dc = self._features_dc
    #     features_rest = self._features_rest
    #     return torch.cat((features_dc, features_rest), dim=1)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        binding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("binding")]
        binding_names = sorted(binding_names, key = lambda x: int(x.split('_')[-1]))
        binding = np.zeros((xyz.shape[0], len(binding_names)), dtype=np.int32)
        for idx, attr_name in enumerate(binding_names):
            binding[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.binding = torch.tensor(binding, dtype=torch.int32, device="cuda").squeeze(-1)

        self.active_sh_degree = self.max_sh_degree
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(1):
            l.append('binding_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        binding = self.binding.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, binding[:, None]), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
