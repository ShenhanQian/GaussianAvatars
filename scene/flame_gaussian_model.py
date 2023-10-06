import torch
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
    
    # TODO: do we need to rotate the SH function?
    # @property
    # def get_features(self):
    #     features_dc = self._features_dc
    #     features_rest = self._features_rest
    #     return torch.cat((features_dc, features_rest), dim=1)
