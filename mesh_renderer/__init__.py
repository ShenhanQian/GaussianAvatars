# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from typing import Tuple, Literal
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import numpy as np
from utils import vector_ops as V
from scene.cameras import MiniCam


class NVDiffRenderer(torch.nn.Module):
    def __init__(
            self,
            use_opengl: bool = False, 
            lighting_type: Literal['constant', 'front'] = 'front',
            lighting_space: Literal['camera', 'world'] = 'camera',
        ):
        super().__init__()
        self.use_opengl = use_opengl
        self.lighting_type = lighting_type
        self.lighting_space = lighting_space
        self.glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    def mvp_from_camera_param(self, RT, K, image_size):
        # projection matrix
        proj = self.projection_from_intrinsics(K, image_size)

        # Modelview and modelview + projection matrices.
        if RT.shape[-2] == 3:
            mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
            mv[..., 3, 3] = 1
        elif RT.shape[-2] == 4:
            mv = RT
        mvp = torch.bmm(proj, mv)
        return mvp
    
    def projection_from_intrinsics(self, K: torch.Tensor, image_size: Tuple[int], near: float=0.1, far:float=10):
        """
        Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: down, z: in)
        Args:
            K: Intrinsic matrix, (N, 3, 3)
                K = [[
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1],
                    ]
                ]
            image_size: (height, width)
        Output:
            proj = [[
                    [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                    [0.0,    2*fy/h, (h - 2*cy)/h,             0.0                     ],
                    [0.0,    0.0,     -(far+near) / (far-near), -2*far*near / (far-near)],
                    [0.0,    0.0,     -1.0,                     0.0                     ]
                ]
            ]
        """

        B = K.shape[0]
        h, w = image_size

        if K.shape[-2:] == (3, 3):
            fx = K[..., 0, 0]
            fy = K[..., 1, 1]
            cx = K[..., 0, 2]
            cy = K[..., 1, 2]
        elif K.shape[-1] == 4:
            fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
        else:
            raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

        proj = torch.zeros([B, 4, 4], device=K.device)
        proj[:, 0, 0]  = fx * 2 / w 
        proj[:, 1, 1]  = fy * 2 / h
        proj[:, 0, 2]  = (w - 2 * cx) / w
        proj[:, 1, 2]  = (h - 2 * cy) / h
        proj[:, 2, 2]  = -(far+near) / (far-near)
        proj[:, 2, 3]  = -2*far*near / (far-near)
        proj[:, 3, 2]  = -1
        return proj
    
    def world_to_camera(self, vtx, RT):
        """Transform vertex positions from the world space to the camera space"""
        RT = torch.from_numpy(RT).cuda() if isinstance(RT, np.ndarray) else RT
        if RT.shape[-2] == 3:
            mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
            mv[..., 3, 3] = 1
        elif RT.shape[-2] == 4:
            mv = RT

        # (x,y,z) -> (x',y',z',w)
        assert vtx.shape[-1] in [3, 4]
        if vtx.shape[-1] == 3:
            posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
        elif vtx.shape[-1] == 4:
            posw = vtx
        else:
            raise ValueError(f"Expected 3D or 4D points but got: {vtx.shape[-1]}")
        return torch.bmm(posw, RT.transpose(-1, -2))
    
    def camera_to_clip(self, vtx, K, image_size):
        """Transform vertex positions from the camera space to the clip space"""
        K = torch.from_numpy(K).cuda() if isinstance(K, np.ndarray) else K
        proj = self.projection_from_intrinsics(K, image_size)
        
        # (x,y,z) -> (x',y',z',w)
        assert vtx.shape[-1] in [3, 4]
        if vtx.shape[-1] == 3:
            posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
        elif vtx.shape[-1] == 4:
            posw = vtx
        else:
            raise ValueError(f"Expected 3D or 4D points but got: {vtx.shape[-1]}")
        return torch.bmm(posw, proj.transpose(-1, -2))
    
    def world_to_clip(self, vtx, RT, K, image_size, mvp=None):
        """Transform vertex positions from the world space to the clip space"""
        if mvp == None:
            mvp = self.mvp_from_camera_param(RT, K, image_size)

        mvp = torch.from_numpy(mvp).cuda() if isinstance(mvp, np.ndarray) else mvp
        # (x,y,z) -> (x',y',z',w)
        posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
        return torch.bmm(posw, mvp.transpose(-1, -2))
    
    def world_to_ndc(self, vtx, RT, K, image_size, flip_y=False):
        """Transform vertex positions from the world space to the NDC space"""
        verts_clip = self.world_to_clip(vtx, RT, K, image_size)
        verts_ndc = verts_clip[:, :, :3] / verts_clip[:, :, 3:]
        if flip_y:
            verts_ndc[:, :, 1] *= -1
        return verts_ndc

    def compute_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[..., i0, :]
        v1 = verts[..., i1, :]
        v2 = verts[..., i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = V.safe_normalize(face_normals)
        return face_normals
    
    def compute_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[..., i0, :]
        v1 = verts[..., i1, :]
        v2 = verts[..., i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        N = verts.shape[0]
        v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
        v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
        v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

        v_normals = torch.where(V.dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
        v_normals = V.safe_normalize(v_normals)
        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_normals))
        return v_normals
    
    def shade(self, normal, lighting_coeff=None):
        if self.lighting_type == 'constant':
            diffuse = torch.ones_like(normal[..., :3])
        elif self.lighting_type == 'front':
            diffuse = torch.clamp(V.dot(normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')), 0.0, 1.0)
        else:
            raise NotImplementedError(f"Unknown lighting type: {self.lighting_type}")
        return diffuse
    
    def render_from_camera(
        self, verts, faces, cam: MiniCam, background_color=[1., 1., 1.],
        face_colors=None,
    ):
        """
        Renders meshes into RGBA images
        """
        world_view_transform = cam.world_view_transform.clone().to(verts)
        world_view_transform[:,1] = -world_view_transform[:,1]
        world_view_transform[:,2] = -world_view_transform[:,2]
        RT = world_view_transform.T[None, ...]
        
        full_proj_transform = cam.full_proj_transform.clone()
        full_proj_transform[:,1] = -full_proj_transform[:,1]
        full_proj = full_proj_transform.T[None, ...].to(verts)

        if self.use_opengl:
            image_size = cam.image_height, cam.image_width
            
            return self.render_mesh(verts, faces, RT, full_proj, image_size, background_color, face_colors)
        else:
            if cam.image_height > 2048 or cam.image_width > 2048:
                image_size = 2048, 2048
            else:
                image_size = int(cam.image_height // 8 * 8), int(cam.image_width // 8 * 8)

            output = self.render_mesh(verts, faces, RT, full_proj, image_size, background_color, face_colors)
            for k, v in output.items():
                output[k] = F.interpolate(v.permute(0, 3, 1, 2), (cam.image_height, cam.image_width), mode='bilinear').permute(0, 2, 3, 1)
            return output
    
    def render_mesh(
        self, verts, faces, RT, full_proj, image_size, background_color=[1., 1., 1.],
        face_colors=None,
    ):
        """
        Renders meshes into RGBA images
        """

        verts_camera = self.world_to_camera(verts, RT)[..., :3]
        verts_clip = self.world_to_clip(verts, None, None, image_size, mvp=full_proj)
        tri = faces.int()
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)

        faces = faces.int()
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()
        face_id = torch.clamp(rast_out[..., -1:].long() - 1, 0)  # (B, W, H, 1)
        W, H = face_id.shape[1:3]
        face_id_ = face_id[:, :, :, None].expand(-1, -1, -1, -1, 3)  # (B, W, H, 1, 1)

        face_normals = self.compute_face_normals(verts_camera, faces)  # (B, F, 3)
        face_normals_ = face_normals[:, None, None, :, :].expand(-1, W, H, -1, -1)  # (B, 1, 1, F, 3)
        normal = torch.gather(face_normals_, -2, face_id_).squeeze(-2) # (B, W, H, 3)

        if face_colors is not None:
            face_colors_ = face_colors[:, None, None, :, :].expand(-1, W, H, -1, -1)  # (B, 1, 1, F, 3)
            albedo = torch.gather(face_colors_, -2, face_id_).squeeze(-2) # (B, W, H, 3)
        else:
            albedo = torch.ones_like(normal)
            
        # ---- shading ----
        diffuse = self.shade(normal)

        rgb = albedo * diffuse
        alpha = fg_mask.float()
        rgba = torch.cat([rgb, alpha], dim=-1)

        # ---- background ----
        if isinstance(background_color, list):
            """Background as a constant color"""
            rgba_bg = torch.tensor(background_color + [0]).to(rgba).expand_as(rgba)  # RGBA
        elif isinstance(background_color, torch.Tensor):
            """Background as a image"""
            rgba_bg = background_color
            rgba_bg = torch.cat([rgba_bg, torch.zeros_like(rgba_bg[..., :1])], dim=-1)  # RGBA
        else:
            raise ValueError(f"Unknown background type: {type(background_color)}")
        rgba_bg = rgba_bg.flip(1)  # opengl camera has y-axis up, needs flipping
        
        normal = torch.where(fg_mask, normal, rgba_bg[..., :3])
        diffuse = torch.where(fg_mask, diffuse, rgba_bg[..., :3])
        rgba = torch.where(fg_mask, rgba, rgba_bg)

        # ---- AA on both RGB and alpha channels ----
        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces.int())
        
        return {
            'albedo': albedo.flip(1),
            'normal': normal.flip(1),
            'diffuse': diffuse.flip(1),
            'rgba': rgba_aa.flip(1),
        }
