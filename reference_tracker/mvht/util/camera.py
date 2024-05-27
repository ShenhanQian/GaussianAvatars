# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from typing import Tuple, Literal
import torch
import torch.nn.functional as F
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def align_cameras_to_axes(
    R: torch.Tensor,
    T: torch.Tensor,
    target_convention: Literal["opengl", "opencv"] = None,
):
    """align the averaged axes of cameras with the world axes.

    Args:
        R: rotation matrix (N, 3, 3)
        T: translation vector (N, 3)
    """
    # The column vectors of R are the basis vectors of each camera.
    # We construct new bases by taking the mean directions of axes, then use Gram-Schmidt
    # process to make them orthonormal
    bases_c2w = gram_schmidt_orthogonalization(R.mean(0))
    if target_convention == "opengl":
        bases_c2w[:, [1, 2]] *= -1  # flip y and z axes
    elif target_convention == "opencv":
        pass
    bases_w2c = bases_c2w.t()

    # convert the camera poses into the new coordinate system
    R = bases_w2c[None, ...] @ R
    T = bases_w2c[None, ...] @ T
    return R, T


def change_camera_coord_convention(camera_coord_conversion: str, R: torch.Tensor):
    if camera_coord_conversion is not None:
        if camera_coord_conversion == "opencv->opengl":
            R[:, :3, [1, 2]] *= -1
        elif camera_coord_conversion == "opencv->pytorch3d":
            R[:, :3, [0, 1]] *= -1
        elif camera_coord_conversion == "opengl->pytorch3d":
            R[:, :3, [0, 2]] *= -1
        else:
            raise ValueError(
                f"Unknown camera coordinate conversion: {camera_coord_conversion}."
            )
    return R


def gram_schmidt_orthogonalization(M: torch.tensor):
    """conducting Gram-Schmidt process to transform column vectors into orthogonal bases

    Args:
        M: An matrix (num_rows, num_cols)
    Return:
        M: An matrix with orthonormal column vectors (num_rows, num_cols)
    """
    num_rows, num_cols = M.shape
    for c in range(1, num_cols):
        M[:, [c - 1, c]] = F.normalize(M[:, [c - 1, c]], p=2, dim=0)
        M[:, [c]] -= M[:, :c] @ (M[:, :c].T @ M[:, [c]])

    M[:, -1] = F.normalize(M[:, -1], p=2, dim=0)
    return M


def projection_from_intrinsics(K: np.ndarray, image_size: Tuple[int], near: float=0.01, far:float=10, flip_y: bool=False, z_sign=-1):
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
                [0.0,    0.0,     z_sign*(far+near) / (far-near), -2*far*near / (far-near)],
                [0.0,    0.0,     z_sign,                     0.0                     ]
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
        # fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
        fx = K[..., [0]]
        fy = K[..., [1]]
        cx = K[..., [2]]
        cy = K[..., [3]]
    else:
        raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

    proj = np.zeros([B, 4, 4])
    proj[:, 0, 0]  = fx * 2 / w 
    proj[:, 1, 1]  = fy * 2 / h
    proj[:, 0, 2]  = (w - 2 * cx) / w
    proj[:, 1, 2]  = (h - 2 * cy) / h
    proj[:, 2, 2]  = z_sign * (far+near) / (far-near)
    proj[:, 2, 3]  = -2*far*near / (far-near)
    proj[:, 3, 2]  = z_sign

    if flip_y:
        proj[:, 1, 1] *= -1
    return proj


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, znear=1e-8, zfar=10, convention: Literal["opengl", "opencv"]="opengl"):
        self.image_width = W
        self.image_height = H
        self.radius_default = r
        self.fovy_default = fovy
        self.znear = znear
        self.zfar = zfar
        self.convention = convention

        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.reset()
    
    def reset(self):
        """ The internal state of the camera is based on the OpenGL convention, but 
            properties are converted to the target convention when queried.
        """
        self.rot = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # OpenGL convention
        self.look_at = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.radius = self.radius_default  # camera distance from center
        self.fovy = self.fovy_default
        if self.convention == "opencv":
            self.z_sign = 1
            self.y_sign = 1
        elif self.convention == "opengl":
            self.z_sign = -1
            self.y_sign = -1
        else:
            raise ValueError(f"Unknown convention: {self.convention}")

    @property
    def fovx(self):
        return self.fovy / self.image_height * self.image_width
    
    @property
    def intrinsics(self):
        focal = self.image_height / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.image_width // 2, self.image_height // 2])
    
    @property
    def projection_matrix(self):
        return projection_from_intrinsics(self.intrinsics[None], (self.image_height, self.image_width), self.znear, self.zfar, z_sign=self.z_sign)[0]
    
    @property
    def world_view_transform(self):
        return np.linalg.inv(self.pose)  # world2cam

    @property
    def full_proj_transform(self):
        return self.projection_matrix @ self.world_view_transform

    @property
    def pose(self):
        # first move camera to radius
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] += self.radius

        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        pose = rot @ pose

        # translate
        pose[:3, 3] -= self.look_at

        if self.convention == "opencv":
            pose[:, [1, 2]] *= -1
        elif self.convention == "opengl":
            pass
        else:
            raise ValueError(f"Unknown convention: {self.convention}")
        return pose

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.3 * dx)
        rotvec_y = side * np.radians(-0.3 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        d = np.array([dx, -dy, dz])  # the y axis is flipped
        self.look_at += 2 * self.rot.as_matrix()[:3, :3] @ d * self.radius / self.image_height * math.tan(np.radians(self.fovy) / 2)
