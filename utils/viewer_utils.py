# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from typing import Tuple, Literal
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path
import os
from dataclasses import dataclass
import dearpygui.dearpygui as dpg


def projection_from_intrinsics(K: np.ndarray, image_size: Tuple[int], near: float=0.01, far:float=10, flip_y: bool=False, z_sign=-1):
    """
    Transform points from camera space (x: right, y: up, z: out) to clip space (x: right, y: up, z: in)
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
    def __init__(self, W, H, r=2, fovy=60, znear=0.01, zfar=10, convention: Literal["opengl", "opencv"]="opengl", save_path='camera.json'):
        self.image_width = W
        self.image_height = H
        self.radius_default = r
        self.fovy_default = fovy
        self.znear = znear
        self.zfar = zfar
        self.convention = convention
        self.save_path = save_path

        self.reset()
        self.load()
    
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
    
    def save(self):
        save_dict = {
            'rotation': self.rot.as_matrix().tolist(),
            'look_at': self.look_at.tolist(),
            'radius': self.radius,
            'fovy': self.fovy,
        }
        with open(self.save_path, "w") as f:
            json.dump(save_dict, f, indent=4)
    
    def clear(self):
        os.remove(self.save_path)
    
    def load(self):
        if not Path(self.save_path).exists():
            return
        with open(self.save_path, "r") as f:
            load_dict = json.load(f)
        self.rot = R.from_matrix(np.array(load_dict['rotation']))
        self.look_at = np.array(load_dict['look_at'])
        self.radius = load_dict['radius']
        self.fovy = load_dict['fovy']

    @property
    def fovx(self):
        focal = self.image_height / (2 * np.tan(np.radians(self.fovy) / 2))
        fovx = 2 * np.arctan(self.image_width / (2 * focal))
        return np.degrees(fovx)
    
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
        # first move camera to (0, 0, radius)
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

    def orbit_x(self, angle_x):
        axis_x = self.rot.as_matrix()[:3, 0]
        rotvec_x = axis_x * angle_x
        self.rot = R.from_rotvec(rotvec_x) * self.rot
    
    def orbit_y(self, angle_y):
        axis_y = self.rot.as_matrix()[:3, 1]
        rotvec_y = axis_y * angle_y
        self.rot = R.from_rotvec(rotvec_y) * self.rot
    
    def orbit_z(self, angle_z):
        axis_z = self.rot.as_matrix()[:3, 2]
        rotvec_z = axis_z * angle_z
        self.rot = R.from_rotvec(rotvec_z) * self.rot

    def trackball(self, p, q, rot_begin=None):
        axis = np.cross(p, q)
        angle = np.arccos(np.dot(p, q))
        rotvec = axis * angle
        if rot_begin is None:
            self.rot = self.rot * R.from_rotvec(rotvec)
        else:
            self.rot = rot_begin * R.from_rotvec(rotvec)

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx=0, dy=0, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        d = np.array([dx, -dy, dz])  # the y axis is flipped
        self.look_at += 2 * self.rot.as_matrix()[:3, :3] @ d * self.radius / self.image_height * math.tan(np.radians(self.fovy) / 2)

@dataclass
class Mini3DViewerConfig:
    W: int = 960
    """GUI width"""
    H: int = 540
    """GUI height"""
    radius: float = 1
    """default GUI camera radius from center"""
    fovy: float = 20
    """default GUI camera fovy"""

class Mini3DViewer:
    def __init__(self, cfg: Mini3DViewerConfig, title='Mini3DViewer'):
        self.cfg = cfg
        
        # viewer settings
        self.W = cfg.W
        self.H = cfg.H
        self.cam = OrbitCamera(self.W, self.H, r=cfg.radius, fovy=cfg.fovy, convention=cfg.cam_convention)

        # buffers for mouse interaction
        self.cursor_x = None
        self.cursor_y = None
        self.cursor_x_prev = None
        self.cursor_y_prev = None
        self.drag_begin_x = None
        self.drag_begin_y = None
        self.drag_button = None

        # status
        self.last_time_fresh = None
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        
        # temporal settings
        self.timestep = 0  # the chosen timestep of the dataset
        self.num_timesteps = 1

        # initialize GUI
        print("Initializing GUI...")

        # disable GLVND patching on Linux to avoid segmentation fault when deleting texture
        import platform
        import os
        if platform.system().upper() == "LINUX":
            os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

        dpg.create_context()
        self.define_gui()
        self.register_callbacks()
        dpg.create_viewport(title=title, width=self.W, height=self.H, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    
    def __del__(self):
        dpg.destroy_context()
    
    def define_gui(self):
        # register texture =================================================================================================
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        # register window ==================================================================================================
        # the window to display the rendered image
        with dpg.window(label="viewer", tag="_canvas_window", width=self.W, height=self.H, no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True, no_resize=True):
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image")
        
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_canvas_window", theme_no_padding)

    def register_callbacks(self):
        def callback_resize(sender, app_data):
            self.W = app_data[0]
            self.H = app_data[1]
            self.cam.image_width = self.W
            self.cam.image_height = self.H
            self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

            # delete and re-add the texture and image
            dpg.delete_item("_texture")
            dpg.delete_item("_image")

            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image", parent="_canvas_window")
            dpg.configure_item("_canvas_window", width=self.W, height=self.H)
            self.need_update = True
        
        def callback_mouse_move(sender, app_data):
            self.cursor_x, self.cursor_y = app_data

            # drag
            if self.drag_begin_x is not None or self.drag_begin_y is not None:
                if self.cursor_x_prev is None or self.cursor_y_prev is None:
                    cursor_x_prev = self.drag_begin_x
                    cursor_y_prev = self.drag_begin_y
                else:
                    cursor_x_prev = self.cursor_x_prev
                    cursor_y_prev = self.cursor_y_prev
                
                # drag with left button
                if self.drag_button is dpg.mvMouseButton_Left:
                    cx = self.W // 2
                    cy = self.H // 2
                    r = min(cx, cy) * 0.9
                    # rotate with trackball: https://raw.org/code/trackball-rotation-using-quaternions/
                    if (self.drag_begin_x - cx)**2 + (self.drag_begin_y - cy)**2 < r**2:
                        px, py = -(self.drag_begin_x - cx)/r, (self.drag_begin_y - cy)/r
                        px2y2 = px**2 + py**2
                        # p = np.array([px, py, np.sqrt(max(1 - px2y2, 0))])
                        p = np.array([px, py, np.sqrt(1e-6+max(1 - px2y2, 0.25/px2y2))])
                        p /= np.linalg.norm(p)

                        qx, qy = -(self.cursor_x - cx)/r, (self.cursor_y - cy)/r
                        qx2y2 = qx**2 + qy**2
                        # q = np.array([qx, qy, np.sqrt(max(1 - qx2y2, 0))])
                        q = np.array([qx, qy, np.sqrt(1e-6+max(1 - qx2y2, 0.25/qx2y2))])
                        q /= np.linalg.norm(q)

                        self.cam.trackball(p, q, rot_begin=self.cam_rot_begin)

                    # rotate around Z axis
                    else:
                        xy_begin = np.array([cursor_x_prev - cx, cursor_y_prev - cy])
                        xy_end = np.array([self.cursor_x - cx, self.cursor_y - cy])
                        angle_z = np.arctan2(xy_end[1], xy_end[0]) - np.arctan2(xy_begin[1], xy_begin[0])
                        self.cam.orbit_z(angle_z)
                
                # drag with middle button
                elif self.drag_button is dpg.mvMouseButton_Middle or self.drag_button is dpg.mvMouseButton_Right:
                    # Pan in X-Y plane
                    self.cam.pan(dx=self.cursor_x - cursor_x_prev, dy=self.cursor_y - cursor_y_prev)
                self.need_update = True
            
            self.cursor_x_prev = self.cursor_x
            self.cursor_y_prev = self.cursor_y

        def callback_mouse_button_down(sender, app_data):
            if not dpg.is_item_hovered("_canvas_window"):
                return
            if self.drag_button != app_data[0]:
                self.drag_begin_x = self.cursor_x
                self.drag_begin_y = self.cursor_y
                self.drag_button = app_data[0]
                self.cam_rot_begin = self.cam.rot
        
        def callback_mouse_release(sender, app_data):
            self.drag_begin_x = None
            self.drag_begin_y = None
            self.drag_button = None
            self.cursor_x_prev = None
            self.cursor_y_prev = None
            self.cam_rot_begin = None

        def callback_mouse_wheel(sender, app_data):
            delta = app_data
            if dpg.is_item_hovered("_canvas_window"):
                self.cam.scale(delta)
                self.need_update = True
        
        def callback_key_press(sender, app_data):
            step = 30
            if sender == '_mvKey_W':
                self.cam.pan(dz=step)
            elif sender == '_mvKey_S':
                self.cam.pan(dz=-step)
            elif sender == '_mvKey_A':
                self.cam.pan(dx=step)
            elif sender == '_mvKey_D':
                self.cam.pan(dx=-step)
            elif sender == '_mvKey_E':
                self.cam.pan(dy=step)
            elif sender == '_mvKey_Q':
                self.cam.pan(dy=-step)

            self.need_update = True

        with dpg.handler_registry():
            dpg.set_viewport_resize_callback(callback_resize)

            # this registry order helps avoid false fire
            dpg.add_mouse_release_handler(callback=callback_mouse_release)
            # dpg.add_mouse_drag_handler(callback=callback_mouse_drag)  # not using the drag callback, since it does not return the starting point
            dpg.add_mouse_move_handler(callback=callback_mouse_move)
            dpg.add_mouse_down_handler(callback=callback_mouse_button_down)
            dpg.add_mouse_wheel_handler(callback=callback_mouse_wheel)

            dpg.add_key_press_handler(dpg.mvKey_W, callback=callback_key_press, tag='_mvKey_W')
            dpg.add_key_press_handler(dpg.mvKey_S, callback=callback_key_press, tag='_mvKey_S')
            dpg.add_key_press_handler(dpg.mvKey_A, callback=callback_key_press, tag='_mvKey_A')
            dpg.add_key_press_handler(dpg.mvKey_D, callback=callback_key_press, tag='_mvKey_D')
            dpg.add_key_press_handler(dpg.mvKey_E, callback=callback_key_press, tag='_mvKey_E')
            dpg.add_key_press_handler(dpg.mvKey_Q, callback=callback_key_press, tag='_mvKey_Q')

