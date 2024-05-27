from typing import Tuple, Literal, Optional
# from pytorch3d.structures.meshes import Meshes
import nvdiffrast.torch as dr
import torch.nn.functional as F
import torch
import numpy as np
from mvht.util import vector_ops as V


def get_SH_shading(normals, sh_coefficients, sh_const):
    """
    :param normals: shape N, H, W, K, 3
    :param sh_coefficients: shape N, 9, 3
    :return:
    """

    N = normals

    # compute sh basis function values of shape [N, H, W, K, 9]
    sh = torch.stack(
        [
            N[..., 0] * 0.0 + 1.0,
            N[..., 0],
            N[..., 1],
            N[..., 2],
            N[..., 0] * N[..., 1],
            N[..., 0] * N[..., 2],
            N[..., 1] * N[..., 2],
            N[..., 0] ** 2 - N[..., 1] ** 2,
            3 * (N[..., 2] ** 2) - 1,
        ],
        dim=-1,
    )
    sh = sh * sh_const[None, None, None, :].to(sh.device)

    # shape [N, H, W, K, 9, 1]
    sh = sh[..., None]

    # shape [N, H, W, K, 9, 3]
    sh_coefficients = sh_coefficients[:, None, None, :, :]

    # shape after linear combination [N, H, W, K, 3]
    shading = torch.sum(sh_coefficients * sh, dim=3)
    return shading


class NVDiffRenderer(torch.nn.Module):
    def __init__(
            self,
            use_opengl: bool = True, 
            lighting_type: Literal['constant', 'front', 'front-range', 'SH'] = 'front',
            lighting_space: Literal['camera', 'world'] = 'world',
            disturb_rate_fg: Optional[float] = 0.5,
            disturb_rate_bg: Optional[float] = 0.5,
            fid2cid: Optional[torch.Tensor] = None,
        ):
        super().__init__()
        self.backend = 'nvdiffrast'
        self.lighting_type = lighting_type
        self.lighting_space = lighting_space
        self.disturb_rate_fg = disturb_rate_fg
        self.disturb_rate_bg = disturb_rate_bg
        self.glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        self.fragment_cache = None

        if fid2cid is not None:
            fid2cid = F.pad(fid2cid, [1, 0], value=0)  # for nvdiffrast, fid==0 means background pixels
            self.register_buffer("fid2cid", fid2cid, persistent=False)

        # constant factor of first three bands of spherical harmonics
        pi = np.pi
        sh_const = torch.tensor(
            [
                1 / np.sqrt(4 * pi),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
            ],
            dtype=torch.float32,
        )
        self.register_buffer("sh_const", sh_const, persistent=False)

    def clear_cache(self):
        self.fragment_cache = None
    
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
    
    def world_to_clip(self, vtx, RT, K, image_size):
        """Transform vertex positions from the world space to the clip space"""
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

    def rasterize(self, verts, faces, RT, K, image_size, use_cache=False, require_grad=False):
        """
        Rasterizes meshes using a standard rasterization approach
        :param meshes:
        :param cameras:
        :param image_size:
        :return: fragments:
                 screen_coords: N x H x W x 2  with x, y values following pytorch3ds NDC-coord system convention
                                top left = +1, +1 ; bottom_right = -1, -1
        """
        # v_normals = self.compute_v_normals(verts, faces)
        # vertices and faces
        verts_camera = self.world_to_camera(verts, RT)
        verts_clip = self.camera_to_clip(verts_camera, K, image_size)
        tri = faces.int()
        rast_out, rast_out_db = self.rasterize_fragments(verts_clip, tri, image_size, use_cache, require_grad)
        rast_dict = {
            "rast_out": rast_out,
            "rast_out_db": rast_out_db,
            "verts": verts,
            "verts_camera": verts_camera[..., :3],
            "verts_clip": verts_clip,
        }
        
        # if not require_grad:
        #     verts_ndc = verts_clip[:, :, :3] / verts_clip[:, :, 3:]
        #     screen_coords = self.compute_screen_coords(rast_out, verts_ndc, faces, image_size)
        #     rast_dict["screen_coords"] = screen_coords

        return rast_dict

    def rasterize_fragments(self, verts_clip, tri, image_size, use_cache, require_grad=False):
        """
        Either rasterizes meshes or returns cached result
        """

        if not use_cache or self.fragment_cache is None:
            if require_grad:
                rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            else:
                with torch.no_grad():
                    rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            self.fragment_cache = (rast_out, rast_out_db)

        return self.fragment_cache

    def compute_screen_coords(self, rast_out: torch.Tensor, verts:torch.Tensor, faces:torch.Tensor, image_size: Tuple[int]):
        """ Compute screen coords for visible pixels
        Args:
            verts: (N, V, 3), the verts should lie in the ndc space 
            faces: (F, 3)
        """
        N = verts.shape[0]
        F = faces.shape[0]
        meshes = Meshes(verts, faces[None, ...].expand(N, -1, -1))
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        face_verts = verts_packed[faces_packed]

        # NOTE: nvdiffrast shifts face index by +1, and use 0 to flag empty pixel
        pix2face = rast_out[..., -1:].long() - 1  # (N, H, W, 1)
        is_visible = pix2face > -1  # (N, H, W, 1)
        # NOTE: is_visible is computed before packing pix2face to ensure correctness
        pix2face_packed = pix2face + torch.arange(0, N)[:, None, None, None].to(pix2face) * F

        bary_coords = rast_out[..., :2]  # (N, H, W, 2)
        bary_coords = torch.cat([bary_coords, 1 - bary_coords.sum(dim=-1, keepdim=True)], dim =-1)  # (N, H, W, 3)

        visible_faces = pix2face_packed[is_visible]  # (sum(is_visible), 3, 3)
        visible_face_verts = face_verts[visible_faces]
        visible_bary_coords = bary_coords[is_visible[..., 0]]  # (sum(is_visible), 3, 1)
        # visible_bary_coords = torch.cat([visible_bary_coords, 1 - visible_bary_coords.sum(dim=-1, keepdim=True)], dim =-1)

        visible_surface_point = visible_face_verts * visible_bary_coords[..., None]
        visible_surface_point = visible_surface_point.sum(dim=1)

        screen_coords = torch.zeros(*pix2face_packed.shape[:3], 2, device=meshes.device)
        screen_coords[is_visible[..., 0]] = visible_surface_point[:, :2]  # now have gradient

        return screen_coords
    
    def compute_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[..., i0, :]
        v1 = verts[..., i1, :]
        v2 = verts[..., i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0)
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
    
    def compute_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[..., i0, :]
        v1 = verts[..., i1, :]
        v2 = verts[..., i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = V.safe_normalize(face_normals)
        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(face_normals))
        return face_normals
    
    def shade(self, normal, lighting_coeff=None):
        if self.lighting_type == 'constant':
            diffuse = torch.ones_like(normal[..., :3])
        elif self.lighting_type == 'front':
            # diffuse = torch.clamp(V.dot(normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')), 0.0, 1.0)
            diffuse = V.dot(normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
            mask_backface = diffuse < 0
            diffuse[mask_backface] = diffuse[mask_backface].abs()*0.3
        elif self.lighting_type == 'front-range':
            bias = 0.75
            diffuse = torch.clamp(V.dot(normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')) + bias, 0.0, 1.0)
        elif self.lighting_type == 'SH':
            diffuse = get_SH_shading(normal, lighting_coeff, self.sh_const)
        else:
            raise NotImplementedError(f"Unknown lighting type: {self.lighting_type}")
        return diffuse
    
    def detach_by_indices(self, x, indices):
        x = x.clone()
        x[:, indices] = x[:, indices].detach()
        return x
    
    def render_rgba(
        self, rast_dict, verts, faces, verts_uv, faces_uv, tex, lights, background_color=[1., 1., 1.],
        align_texture_except_fid=None, align_boundary_except_vid=None,
    ):
        """
        Renders flame RGBA images
        """

        rast_out = rast_dict["rast_out"]
        rast_out_db = rast_dict["rast_out_db"]
        verts = rast_dict["verts"]
        verts_camera = rast_dict["verts_camera"]
        verts_clip = rast_dict["verts_clip"]
        faces = faces.int()
        faces_uv = faces_uv.int()
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()

        out_dict = {}

        # ---- vertex attributes ----
        if  self.lighting_space == 'world':
            v_normal = self.compute_v_normals(verts, faces)
        elif  self.lighting_space == 'camera':
            v_normal = self.compute_v_normals(verts_camera, faces)
        else:
            raise NotImplementedError(f"Unknown lighting space: {self.lighting_space}")

        v_attr = [v_normal]
     
        v_attr = torch.cat(v_attr, dim=-1)
        attr, _ = dr.interpolate(v_attr, rast_out, faces)
        normal = attr[..., :3]
        normal = V.safe_normalize(normal)

        # ---- uv-space attributes ----
        texc, texd = dr.interpolate(verts_uv[None, ...], rast_out, faces_uv, rast_db=rast_out_db, diff_attrs='all')
        if align_texture_except_fid is not None:  # TODO: rethink when shading with normal
            fid = rast_out[..., -1:].long()  # the face index is shifted by +1
            mask = torch.zeros(faces.shape[0]+1, dtype=torch.bool, device=fid.device)
            mask[align_texture_except_fid + 1] = True
            b, h, w = rast_out.shape[:3]
            rast_mask = torch.gather(mask.reshape(1, 1, 1, -1).expand(b, h, w, -1), 3, fid)
            texc = torch.where(rast_mask, texc.detach(), texc)

        tex = tex.permute(0, 2, 3, 1).contiguous()  # (N, T, T, 4)
        albedo = dr.texture(tex, texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=None)
        
        # ---- shading ----
        diffuse = self.shade(normal, lights)
        diffuse_detach_normal = self.shade(normal.detach(), lights)

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
        
        rgba = torch.where(fg_mask, rgba, rgba_bg)

        # ---- AA on both RGB and alpha channels ----
        if align_boundary_except_vid is not None:
            verts_clip = self.detach_by_indices(verts_clip, align_boundary_except_vid)
        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, faces.int())
        aa = ((rgba - rgba_aa) != 0).any(dim=-1, keepdim=True).repeat_interleave(4, dim=-1)

        # rgba = torch.masked_scatter(rgba, aa, rgba_aa[aa])
        
        # ---- AA only on RGB channels ----
        # rgb = rgba[..., :3].contiguous()
        # alpha = rgba[..., 3:]
        # rgb = dr.antialias(rgb, rast_out, verts_clip, faces.int())
        # rgba = torch.cat([rgb, alpha], dim=-1)
        
        out_dict.update({
            'albedo': albedo.flip(1),
            'normal': normal.flip(1),
            'diffuse': diffuse.flip(1),
            'diffuse_detach_normal': diffuse_detach_normal.flip(1),
            'rgba': rgba_aa.flip(1),
            'aa': aa[..., :3].float().flip(1),
        })
        return out_dict
    
    def render_without_texture(
        self, verts, faces, RT, K, image_size, background_color=[1., 1., 1.],
    ):
        """
        Renders meshes into RGBA images
        """

        verts_camera_ = self.world_to_camera(verts, RT)
        verts_camera = verts_camera_[..., :3]
        verts_clip = self.camera_to_clip(verts_camera_, K, image_size)
        tri = faces.int()
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)

        faces = faces.int()
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()
        face_id = torch.clamp(rast_out[..., -1:].long() - 1, 0)  # (B, W, H, 1)
        W, H = face_id.shape[1:3]

        face_normals = self.compute_face_normals(verts_camera, faces)  # (B, F, 3)
        face_normals_ = face_normals[:, None, None, :, :].expand(-1, W, H, -1, -1)  # (B, 1, 1, F, 3)
        face_id_ = face_id[:, :, :, None].expand(-1, -1, -1, -1, 3)  # (B, W, H, 1, 1)
        normal = torch.gather(face_normals_, -2, face_id_).squeeze(-2) # (B, W, H, 3)

        albedo = torch.ones_like(normal)
        
        # ---- shading ----
        diffuse = self.shade(normal)

        rgb = albedo * diffuse
        alpha = fg_mask.float()
        rgba = torch.cat([rgb, alpha], dim=-1)

        # ---- background ----
        if isinstance(background_color, list) or isinstance(background_color, tuple):
            """Background as a constant color"""
            rgba_bg = torch.tensor(list(background_color) + [0]).to(rgba).expand_as(rgba)  # RGBA
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
            'verts_clip': verts_clip,
        }

    def render_v_color(
        self, verts, v_color, faces, RT, K, image_size, background_color=[1., 1., 1.],
    ):
        """
        Renders meshes into RGBA images
        """

        verts_camera_ = self.world_to_camera(verts, RT)
        verts_camera = verts_camera_[..., :3]
        verts_clip = self.camera_to_clip(verts_camera_, K, image_size)
        tri = faces.int()
        rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)

        faces = faces.int()
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()
        face_id = torch.clamp(rast_out[..., -1:].long() - 1, 0)  # (B, W, H, 1)
        W, H = face_id.shape[1:3]

        face_normals = self.compute_face_normals(verts_camera, faces)  # (B, F, 3)
        face_normals_ = face_normals[:, None, None, :, :].expand(-1, W, H, -1, -1)  # (B, 1, 1, F, 3)
        face_id_ = face_id[:, :, :, None].expand(-1, -1, -1, -1, 3)  # (B, W, H, 1, 1)
        normal = torch.gather(face_normals_, -2, face_id_).squeeze(-2) # (B, W, H, 3)

        albedo = torch.ones_like(normal)

        v_attr = [v_color]
        v_attr = torch.cat(v_attr, dim=-1)
        attr, _ = dr.interpolate(v_attr, rast_out, faces)
        albedo = attr[..., :3]
        
        # ---- shading ----
        diffuse = self.shade(normal)

        rgb = albedo * diffuse
        alpha = fg_mask.float()
        rgba = torch.cat([rgb, alpha], dim=-1)

        # ---- background ----
        if isinstance(background_color, list) or isinstance(background_color, tuple):
            """Background as a constant color"""
            rgba_bg = torch.tensor(list(background_color) + [0]).to(rgba).expand_as(rgba)  # RGBA
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
