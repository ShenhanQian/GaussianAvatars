import tyro
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path
import time
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from PIL import Image

from utils.viewer_utils import OrbitCamera
from gaussian_renderer import GaussianModel, FlameGaussianModel
from gaussian_renderer import render
from mesh_renderer import NVDiffRenderer


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config:
    pipeline: PipelineConfig
    """Pipeline settings for gaussian splatting rendering"""
    point_path: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    render_mode: Literal['rgb', 'depth', 'opacity'] = 'rgb'
    """NeRF rendering mode"""
    W: int = 1440
    """GUI width"""
    H: int = 1440
    """GUI height"""
    radius: float = 1
    """default GUI camera radius from center"""
    fovy: float = 20
    """default GUI camera fovy"""
    background_color: tuple[float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""

class GaussianSplattingViewer:
    def __init__(self, cfg: Config):
        self.cfg = cfg  # shared with the trainer's cfg to support in-place modification of rendering parameters.
        
        # viewer settings
        self.W = cfg.W
        self.H = cfg.H
        self.cam = OrbitCamera(self.W, self.H, r=cfg.radius, fovy=cfg.fovy, convention="opencv")
        # self.mesh_color = torch.tensor([0.2, 0.5, 1], dtype=torch.float32)  # default white bg
        # self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.last_time_fresh = None
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.debug = True

        # buffers for mouse interaction
        self.cursor_x = None
        self.cursor_y = None
        self.drag_begin_x = None
        self.drag_begin_y = None
        self.drag_button = None

        # rendering settings
        self.render_mode = cfg.render_mode
        self.scaling_modifier: float = 1
        self.num_timesteps = 1
        self.timestep = 0
        self.show_spatting = True
        self.show_mesh = False
        self.mesh_color = torch.tensor([1, 1, 1, 0.5])

        self.define_gui()

    def __del__(self):
        dpg.destroy_context()
    
    def refresh(self):
        dpg.set_value("_texture", self.render_buffer)

        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{fps:.1f}')
        self.last_time_fresh = time.time()

    def define_gui(self):
        dpg.create_context()
        
        # register texture =================================================================================================
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        # register window ==================================================================================================
        # the window to display the rendered image
        with dpg.window(label="viewer", tag="_render_window", width=self.W, height=self.H, no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True, no_resize=True):
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image")

        # control window ==================================================================================================
        with dpg.window(label="Control", tag="_control_window", autosize=True):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS: ")
                dpg.add_text("", tag="_log_fps")

        #     # rendering options
            with dpg.collapsing_header(label="Render", default_open=True):

        #         # render_mode combo
        #         def callback_change_mode(sender, app_data):
        #             self.render_mode = app_data
        #             self.need_update = True
        #         dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

                with dpg.group(horizontal=True):
                    # show nerf
                    def callback_show_splatting(sender, app_data):
                        self.show_spatting = app_data
                        self.need_update = True
                    dpg.add_checkbox(label="show splatting", default_value=self.show_spatting, callback=callback_show_splatting)

                with dpg.group(horizontal=True):
                    # show mesh
                    def callback_show_mesh(sender, app_data):
                        self.show_mesh = app_data
                        self.need_update = True
                    dpg.add_checkbox(label="show mesh", default_value=self.show_mesh, callback=callback_show_mesh)

        #             # show original mesh
        #             def callback_original_mesh(sender, app_data):
        #                 self.original_mesh = app_data
        #                 self.need_update = True
        #             dpg.add_checkbox(label="original mesh", default_value=self.original_mesh, callback=callback_original_mesh)
                
                # timestep slider and buttons
                if self.num_timesteps != None:
                    def callback_set_current_frame(sender, app_data):
                        if sender == "_slider_timestep":
                            self.timestep = app_data
                        elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
                            self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                        elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
                            self.timestep = max(self.timestep - 1, 0)
                        elif sender == "_mvKey_Home":
                            self.timestep = 0
                        elif sender == "_mvKey_End":
                            self.timestep = self.num_timesteps - 1

                        dpg.set_value("_slider_timestep", self.timestep)
                        self.gaussians.select_mesh_by_timestep(self.timestep)

                        self.need_update = True
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                        dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
                        dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=180, min_value=0, max_value=self.num_timesteps - 1, format="%d", default_value=0, callback=callback_set_current_frame)

                # scaling_modifier slider
                def callback_set_scaling_modifier(sender, app_data):
                    self.scaling_modifier = app_data
                    self.need_update = True
                dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=1, format="%.2f", default_value=self.scaling_modifier, callback=callback_set_scaling_modifier, tag="_slider_scaling_modifier")
                
                # mesh_color picker
                def callback_change_mesh_color(sender, app_data):
                    self.mesh_color = torch.tensor(app_data, dtype=torch.float32)  # only need RGB in [0, 1]
                    self.need_update = True
                dpg.add_color_edit((self.mesh_color*255).tolist(), label="Mesh Color", width=200, callback=callback_change_mesh_color)

        #         # bg_color picker
        #         def callback_change_bg(sender, app_data):
        #             self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
        #             self.need_update = True
        #         dpg.add_color_edit((self.bg_color*255).tolist(), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

                # # near slider
                # def callback_set_near(sender, app_data):
                #     self.cam.znear = app_data
                #     self.need_update = True
                # dpg.add_slider_int(label="near", min_value=1e-8, max_value=2, format="%.2f", default_value=self.cam.znear, callback=callback_set_near, tag="_slider_near")

                # # far slider
                # def callback_set_far(sender, app_data):
                #     self.cam.zfar = app_data
                #     self.need_update = True
                # dpg.add_slider_int(label="far", min_value=1e-3, max_value=10, format="%.2f", default_value=self.cam.zfar, callback=callback_set_far, tag="_slider_far")
                
                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True
                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy")

                with dpg.group(horizontal=True):
                    def callback_reset_camera(sender, app_data):
                        self.cam.reset()
                        self.need_update = True
                        if self.debug:
                            dpg.set_value("_log_pose", str(self.cam.pose.astype(np.float16)))
                        dpg.set_value("_slider_fovy", self.cam.fovy)
                    dpg.add_button(label="reset camera", tag="_button_reset_pose", callback=callback_reset_camera)
                    
                    def callback_save_camera(sender, app_data):
                        self.cam.save()
                    dpg.add_button(label="save camera", tag="_button_save_pose", callback=callback_save_camera)

                    def callback_clear_camera(sender, app_data):
                        self.cam.clear()
                    dpg.add_button(label="clear camera", tag="_button_clear_pose", callback=callback_clear_camera)

                with dpg.collapsing_header(label="Camera Pose", default_open=False):
                    dpg.add_text(str(self.cam.pose.astype(np.float16)), tag="_log_pose")
            
                dpg.add_separator()
                def callback_save_image(sender, app_data):
                    if not self.cfg.save_folder.exists():
                        self.cfg.save_folder.mkdir(parents=True)
                    path = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                    print(f"Saving image to {path}")
                    Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="save image", tag="_button_save_image", callback=callback_save_image)

        ### register mouse handlers ========================================================================================

        def callback_mouse_move(sender, app_data):
            self.cursor_x, self.cursor_y = app_data
            if not dpg.is_item_focused("_render_window"):
                return

            if self.drag_begin_x is None or self.drag_begin_y is None:
                self.drag_begin_x = self.cursor_x
                self.drag_begin_y = self.cursor_y
            else:
                dx = self.cursor_x - self.drag_begin_x
                dy = self.cursor_y - self.drag_begin_y

                # button=dpg.mvMouseButton_Left
                if self.drag_button is dpg.mvMouseButton_Left:
                    self.cam.orbit(dx, dy)
                    self.need_update = True
                elif self.drag_button is dpg.mvMouseButton_Middle:
                    self.cam.pan(dx, dy)
                    self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose.astype(np.float16)))

        def callback_mouse_button_down(sender, app_data):
            if not dpg.is_item_focused("_render_window"):
                return
            self.drag_begin_x = self.cursor_x
            self.drag_begin_y = self.cursor_y
            self.drag_button = app_data[0]
        
        def callback_mouse_release(sender, app_data):
            self.drag_begin_x = None
            self.drag_begin_y = None
            self.drag_button = None

            self.dx_prev = None
            self.dy_prev = None
        
        def callback_mouse_drag(sender, app_data):
            if not dpg.is_item_focused("_render_window"):
                return

            button, dx, dy = app_data
            if self.dx_prev is None or self.dy_prev is None:
                ddx = dx
                ddy = dy
            else:
                ddx = dx - self.dx_prev
                ddy = dy - self.dy_prev

            self.dx_prev = dx
            self.dy_prev = dy

            if ddx != 0 and ddy != 0:
                if button is dpg.mvMouseButton_Left:
                    self.cam.orbit(ddx, ddy)
                    self.need_update = True
                elif button is dpg.mvMouseButton_Middle:
                    self.cam.pan(ddx, ddy)
                    self.need_update = True

        def callbackmouse_wheel(sender, app_data):
            delta = app_data
            if dpg.is_item_focused("_render_window"):
                self.cam.scale(delta)
                self.need_update = True
                if self.debug:
                    dpg.set_value("_log_pose", str(self.cam.pose.astype(np.float16)))
            else:
                self.timestep = min(max(self.timestep - delta, 0), self.num_timesteps - 1)
                dpg.set_value("_slider_timestep", self.timestep)
                self.gaussians.select_mesh_by_timestep(self.timestep)
                self.need_update = True

        with dpg.handler_registry():
            # this registry order helps avoid false fire
            dpg.add_mouse_release_handler(callback=callback_mouse_release)
            # dpg.add_mouse_drag_handler(callback=callback_mouse_drag)  # not using the drag callback, since it does not return the starting point
            dpg.add_mouse_move_handler(callback=callback_mouse_move)
            dpg.add_mouse_down_handler(callback=callback_mouse_button_down)
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel)

            # key press handlers
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

        def callback_viewport_resize(sender, app_data):
            while self.rendering:
                time.sleep(0.01)
            self.need_update = False
            self.W = app_data[0]
            self.H = app_data[1]
            self.cam.image_width = self.W
            self.cam.image_height = self.H
            self.render_buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)

            # delete and re-add the texture and image
            dpg.delete_item("_texture")
            dpg.delete_item("_image")

            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image", parent="_render_window")
            dpg.configure_item("_render_window", width=self.W, height=self.H)
            self.need_update = True
        dpg.set_viewport_resize_callback(callback_viewport_resize)

        ### global theme ==================================================================================================
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_render_window", theme_no_padding)

        ### finish setup ==================================================================================================
        dpg.create_viewport(title='Gaussian Splatting Viewer - Local', width=self.W, height=self.H, resizable=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    @torch.no_grad()
    def run(self):
        # self.gaussians = GaussianModel(self.cfg.sh_degree)
        self.gaussians = FlameGaussianModel(self.cfg.sh_degree)

        mesh_renderer = NVDiffRenderer(use_opengl=False)

        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['left_half'])
        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['right_half'])
        # unselected_fid = self.gaussians.flame_model.mask.get_fid_except_fids(selected_fid)
        unselected_fid = []
        
        if self.cfg.point_path is not None:
            if self.cfg.point_path.exists():
                self.gaussians.load_ply(self.cfg.point_path, has_target=False, motion_path=self.cfg.motion_path, disable_fid=unselected_fid)
            else:
                raise FileNotFoundError(f'{self.cfg.point_path} does not exist.')
        
        if self.gaussians.binding != None:
            self.num_timesteps = self.gaussians.num_timesteps
            dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)

            self.gaussians.select_mesh_by_timestep(self.timestep)
        
        faces = self.gaussians.faces.clone()
        # faces = faces[selected_fid]
        
        while dpg.is_dearpygui_running():

            if self.need_update:
                self.rendering = True
                cam = self.prepare_camera()

                if self.show_spatting:
                    rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, torch.tensor(self.cfg.background_color).cuda(), scaling_modifier=self.scaling_modifier)["render"].permute(1, 2, 0).contiguous()

                if self.gaussians.binding != None and self.show_mesh:
                    out_dict = mesh_renderer.render_from_camera(self.gaussians.verts, faces, cam)

                    rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                    rgb_mesh = rgba_mesh[:, :, :3]
                    alpha_mesh = rgba_mesh[:, :, 3:]

                    mesh_opacity = self.mesh_color[3:].cuda()
                    mesh_color = self.mesh_color[:3].cuda()
                    rgb_mesh = rgb_mesh * (alpha_mesh * mesh_color * mesh_opacity + (1 - alpha_mesh))

                if self.show_spatting and self.show_mesh:
                    rgb = rgb_mesh * alpha_mesh  + rgb_splatting * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                elif self.show_spatting and not self.show_mesh:
                    rgb = rgb_splatting
                elif not self.show_spatting and self.show_mesh:
                    rgb = rgb_mesh
                else:
                    rgb = torch.ones([self.H, self.W, 3])

                self.render_buffer = rgb.cpu().numpy()
                self.refresh()
                self.rendering = False
                self.need_update = False
            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = GaussianSplattingViewer(cfg)
    gui.run()
