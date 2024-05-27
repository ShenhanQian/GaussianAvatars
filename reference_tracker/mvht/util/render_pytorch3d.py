from pytorch3d.renderer import BlendParams, PerspectiveCameras, rasterize_meshes
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.blending import sigmoid_alpha_blend, softmax_rgb_blend
from pytorch3d.structures.meshes import Meshes

import torch.nn.functional as F
import torch
import numpy as np


def create_camera_objects(K, RT, resolution, device):
    """
    Create pytorch3D camera objects from camera parameters
    :param K:
    :param RT:
    :param resolution:
    :return:
    """
    R = RT[..., :3]
    T = RT[..., 3]
    H, W = resolution
    img_size = torch.tensor([[H, W]] * len(K), dtype=torch.int, device=device)
    f = torch.stack((K[..., 0, 0], K[..., 1, 1]), dim=-1)
    principal_point = torch.cat([K[..., [0], -1], K[..., [1], -1]], dim=1)
    cameras = PerspectiveCameras(
        R=R.transpose(-1, -2),  # conversion from OpenCV to PyTorch3D
        T=T,
        principal_point=principal_point,
        focal_length=f,
        device=device,
        image_size=img_size,
        in_ndc=False,
    )
    return cameras


def hard_feature_blend(features, fragments, blend_params) -> torch.Tensor:
    """
    Naive blending of top K faces to return a feature image
      - **D** - choose features of the closest point i.e. K=0
      - **A** - is_background --> pix_to_face == -1

    Args:
        features: (N, H, W, K, D) features for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
        blend_params: BlendParams instance that contains a background_color
        field specifying the color for the background
    Returns:
        RGBA pixel_colors: (N, H, W, 4)
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    # Mask for the background.
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)

    if torch.is_tensor(blend_params.background_color):
        background_color = blend_params.background_color.to(device)
    else:
        background_color = features.new_tensor(blend_params.background_color)  # (3)

    # Find out how much background_color needs to be expanded to be used for masked_scatter.
    num_background_pixels = is_background.sum()

    # Set background color.
    pixel_features = features[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, 3)

    # Concat with the alpha channel.
    alpha = (~is_background).float()[..., None]
    return torch.cat([pixel_features, alpha], dim=-1)  # (N, H, W, 4)


def get_SH_shading(per_face_normals, sh_coefficients, sh_const):
    """
    :param per_face_normals: shape N, H, W, K, 3
    :param sh_coefficients: shape N, 9, 3
    :return:
    """

    N = per_face_normals

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
    sh = sh * sh_const[None, None, None, None, :].to(sh.device)

    # shape [N, H, W, K, 9, 1]
    sh = sh[..., None]

    # shape [N, H, W, K, 9, 3]
    sh_coefficients = sh_coefficients[:, None, None, None, :, :]

    # shape after linear combination [N, H, W, K, 3]
    shading = torch.sum(sh_coefficients * sh, dim=4)
    return shading


class PyTorch3DRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = 'pytorch3d'
        self.fragment_cache = None
        pi = np.pi

        # constant factor of first three bands of spherical harmonics
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

    def world_to_ndc(self, pos, RT, K, image_size, flip_xy=True):
        """ Transform vertex positions from the world space to the NDC space
            NOTE: this method behaves differently from cameras.transform_points_ndc
        """
        cameras = create_camera_objects(K, RT, image_size, device=K.device)
        pos_ndc = cameras.transform_points_ndc(pos)

        H, W = image_size
        if H > W:
            s = H / W
            pos_ndc[..., 1] *= 1 / s
        elif H < W:
            s = W / H
            pos_ndc[..., 0] *= 1 / s

        if flip_xy:
            pos_ndc[..., :2] = -pos_ndc[..., :2]
        return pos_ndc

    def clear_cache(self):
        self.fragment_cache = None

    def rasterize(self, verts, faces, RT, K, image_size, use_cache):
        """
        Rasterizes meshes using a standard rasterization approach
        :param meshes:
        :param cameras:
        :param image_size:
        :return: fragments:
                 screen_coords: N x H x W x 2  with x, y values following pytorch3ds NDC-coord system convention
                                top left = +1, +1 ; bottom_right = -1, -1
        """
        cameras = create_camera_objects(K, RT, image_size, device=verts.device)

        # create mesh from selected faces and vertices
        verts = verts.expand(len(K), -1, -1)
        faces = faces.expand(len(verts), -1, -1)
        meshes = Meshes(verts=verts, faces=faces)

        assert len(meshes) == len(cameras)

        eps = None
        verts_world = meshes.verts_padded()
        verts_view = cameras.get_world_to_view_transform().transform_points(
            verts_world, eps=eps
        )
        projection_trafo = cameras.get_projection_transform().compose(
            cameras.get_ndc_camera_transform()
        )
        verts_ndc = projection_trafo.transform_points(verts_view, eps=eps)
        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes.update_padded(new_verts_padded=verts_ndc)

        fragments = self.rasterize_fragments(
            cameras, meshes_ndc, image_size, use_cache
        )
        screen_coords = self.compute_screen_coords(meshes_ndc, fragments, image_size)
        return {"fragments": fragments, "screen_coords": screen_coords}
    

    def rasterize_fragments(self, cameras, meshes_ndc, image_size, use_cache):
        """
        Either rasterizes meshes_ndc or returns cached result
        """

        znear = cameras.get_znear()
        if isinstance(znear, torch.Tensor):
            znear = znear.min().item()
        z_clip = None if znear is None else znear / 2

        if not use_cache or self.fragment_cache is None:
            with torch.no_grad():
                fragments = rasterize_meshes(
                    meshes_ndc,
                    image_size=image_size,
                    blur_radius=0,
                    faces_per_pixel=1,
                    bin_size=None,
                    max_faces_per_bin=None,
                    clip_barycentric_coords=False,
                    perspective_correct=True,
                    cull_backfaces=True,
                    z_clip_value=z_clip,
                    cull_to_frustum=False,
                )

                self.fragment_cache = Fragments(
                    pix_to_face=fragments[0],
                    zbuf=fragments[1],
                    bary_coords=fragments[2],
                    dists=fragments[3],
                )

        return self.fragment_cache

    def compute_screen_coords(self, meshes_ndc, fragments, image_size):
        verts_packed = meshes_ndc.verts_packed()
        faces_packed = meshes_ndc.faces_packed()
        face_verts = verts_packed[faces_packed]

        # right now this only works with faces_per_pixel == 1
        pix2face, bary_coords = fragments.pix_to_face, fragments.bary_coords
        is_visible = pix2face[..., 0] > -1

        # shape (sum(is_visible), 3, 3)
        visible_faces = pix2face[is_visible][:, 0]
        visible_face_verts = face_verts[visible_faces]
        
        # shape (sum(is_visible), 3, 1)
        visible_bary_coords = bary_coords[is_visible][:, 0, :, None]
        visible_surface_point = (visible_face_verts * visible_bary_coords).sum(dim=1)

        screen_coords = torch.zeros(*pix2face.shape[:3], 2, device=meshes_ndc.device)
        screen_coords[is_visible] = visible_surface_point[:, :2]  # now have gradient

        # if images are not-squared we need to adjust the screen coordinates
        # by the aspect ratio => coords given as [-1,1] for shorter edge and
        # [-s,s] for longer edge where s is the aspect ratio
        H, W = image_size
        if H > W:
            s = H / W
            screen_coords[..., 1] *= 1 / s
        elif H < W:
            s = W / H
            screen_coords[..., 0] *= 1 / s

        # NOTE: PyTorch3D filps x/y direction in the NDC space by default
        return screen_coords * -1

    def render_rgba(
        self, rast_dict, verts, faces, uv_coords, faces_uv, albedos, lights, 
        background_color=[0, 0, 0], softblend=False
    ):
        """
        Renders flame RGBA images
        """
        fragments = rast_dict["fragments"]

        # compute verts_uv coordinates for each face at each pixel
        N, H, W, K_faces, _ = fragments.bary_coords.shape

        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, uv_coords
        )

        # pixel_uvs: (N, H, W, K_faces, 3) -> (N, K_faces, H, W, 3) -> (NK_faces, H, W, 3)
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K_faces, H, W, 3)

        # obtain texture color from the respective textures and verts_uv maps
        tex_stack = torch.cat(
            [albedos[[i]].expand(K_faces, -1, -1, -1) for i in range(N)]
        )
        head = F.grid_sample(tex_stack, pixel_uvs[..., :2], align_corners=False)

        # features shape N*K_faces x C x H x W -> N, H, W, K_faces, C
        features = head.reshape(N, K_faces, -1, H, W).permute(0, 3, 4, 1, 2)

        # NORMAL RENDERING
        # mesh = Meshes(verts=verts, faces=faces[None, ...].expand(verts.shape[0], -1, -1))
        # face_normals = mesh.verts_normals_packed()[mesh.faces_packed()]
        # # ATTENTION: NORMALS ARE NOT TRANSFORMED IN CAM SPACE YET
        # pixel_face_normals = interpolate_face_attributes(
        #     fragments.pix_to_face, fragments.bary_coords, face_normals
        # )  # N x H x W x K_faces x 3

        # # LIGHTING
        # shading = get_SH_shading(pixel_face_normals, lights, self.sh_const)
        # features = ((features * 0.5 + 0.5) * shading) * 2 - 1

        if softblend:
            blend_params = BlendParams(sigma=1e-8, gamma=1e-8, background_color=background_color)
            feature_img = softmax_rgb_blend(features, fragments, blend_params)
        else:
            blend_params = BlendParams(sigma=0, gamma=0, background_color=background_color)
            feature_img = hard_feature_blend(features, fragments, blend_params)

        return feature_img

    def render_normal(self, rast_dict, verts, faces, uv_coords, background_color=[0, 0, 0]):
        """
        Renders flame normal images
        """
        fragments = rast_dict["fragments"]

        # compute verts_uv coordinates for each face at each pixel
        N, H, W, K_faces, _ = fragments.bary_coords.shape

        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, uv_coords
        )

        # pixel_uvs: (N, H, W, K_faces, 3) -> (N, K_faces, H, W, 3) -> (NK_faces, H, W, 3)
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K_faces, H, W, 3)

        # NORMAL RENDERING
        mesh = Meshes(verts=verts, faces=faces[None, ...].expand(verts.shape[0], -1, -1))
        face_normals = mesh.verts_normals_packed()[mesh.faces_packed()]
        face_normals = face_normals / 2 + 0.5

        # ATTENTION: NORMALS ARE NOT TRANSFORMED IN CAM SPACE YET
        pixel_face_normals = interpolate_face_attributes(
            fragments.pix_to_face,
            torch.ones_like(fragments.bary_coords) / 3,
            face_normals,
        )  # N x H x W x K_faces x 3

        features = pixel_face_normals

        blend_params = BlendParams(sigma=0, gamma=0, background_color=background_color)
        feature_img = hard_feature_blend(features, fragments, blend_params)

        return feature_img.permute(0, 3, 1, 2)
