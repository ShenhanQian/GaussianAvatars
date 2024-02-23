# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import tyro
from dataclasses import dataclass
from typing import Literal
import socket
import json
import numpy as np
import time
import dearpygui.dearpygui as dpg
import numpy as np
# import torch.nn
# import torch.nn.functional as F

from utils.viewer_utils import OrbitCamera


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 6009
    pause_rendering: bool = False
    render_mode: Literal['rgb', 'depth', 'opacity'] = 'rgb'
    """NeRF rendering mode"""
    show_splatting: bool = True
    """show NeRF rendering"""
    show_mesh: bool = False
    """show mesh rendering"""
    training: bool = True
    """start training at launch time"""
    W: int = 960
    """GUI width"""
    H: int = 540
    """GUI height"""
    radius: float = 1
    """default GUI camera radius from center"""
    fovy: float = 20
    """default GUI camera fovy"""


class RemoteViewer:
    def __init__(self, cfg: Config):
        self.cfg = cfg  # shared with the trainer's cfg to support in-place modification of rendering parameters.
        
        # viewer settings
        self.W = cfg.W
        self.H = cfg.H
        self.cam = OrbitCamera(self.W, self.H, r=cfg.radius, fovy=cfg.fovy, convention="opengl")
        # self.mesh_color = torch.tensor([0.2, 0.5, 1], dtype=torch.float32)  # default white bg
        # self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.training = cfg.training
        self.pause_rendering = cfg.pause_rendering
        self.step = 0  # training step
        self.timestamp_begin = time.time() if self.training else None
        self.elapsed_last = 0  # training time
        self.last_time_fresh = None
        self.timestep = 0  # the chosen timestep of the dataset
        self.num_timesteps = 1
        self.num_points = 0
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.debug = True

        # buffers for mouse interaction
        self.cursor_x = None
        self.cursor_y = None
        self.drag_begin_x = None
        self.drag_begin_y = None
        self.drag_button = None

        # gaussian splatting rendering settings
        self.show_splatting = cfg.show_splatting
        self.render_mode = cfg.render_mode
        self.canonical_space = False
        self.scaling_modifier: float = 1
        self.use_original_mesh: bool = False

        # mesh rendering settings
        self.show_mesh = cfg.show_mesh
        self.mesh_opacity = 0.5
        self.original_mesh = False

        # network
        self.socket = None

        dpg.create_context()
        self.register_dpg()

    def __del__(self):
        dpg.destroy_context()
    
    def send_message(self, message_bytes, message_length):
        self.socket.sendall(message_length)
        self.socket.sendall(message_bytes)
    
    def send_json(self):
        if self.pause_rendering:
            message = {
                "resolution_x": 0,
                "resolution_y": 0,
                "do_training": self.training,
                "keep_alive": True,
            }
        else:
            message = {
                "resolution_x": self.cam.image_width,
                "resolution_y": self.cam.image_height,
                "do_training": self.training,
                "fov_y": np.radians(self.cam.fovy),
                "fov_x": np.radians(self.cam.fovx),
                "z_near": self.cam.znear,
                "z_far": self.cam.zfar,
                "keep_alive": True,
                "scaling_modifier": self.scaling_modifier,
                "show_splatting": self.show_splatting,
                "show_mesh": self.show_mesh,
                "mesh_opacity": self.mesh_opacity,
                "use_original_mesh": self.use_original_mesh,
                "view_matrix": self.cam.world_view_transform.T.flatten().tolist(),  # the transpose is required by gaussian splatting rasterizer
                "view_projection_matrix": self.cam.full_proj_transform.T.flatten().tolist(),  # the transpose is required by gaussian splatting rasterizer
                'timestep': self.timestep,
            }
        message_str = json.dumps(message)

        message_bytes = message_str.encode("utf-8")
        message_length = len(message_bytes).to_bytes(4, 'little')
        self.send_message(message_bytes, message_length)
    
    def receive_bytes(self, bytes_expected):
        bytes_received = 0

        chunks = []
        while bytes_received < bytes_expected:
            chunk = self.socket.recv(min(bytes_expected - bytes_received, 4096))
            if not chunk:
                break  # Connection closed
            chunks.append(chunk)
            bytes_received += len(chunk)

        buffer = b''.join(chunks)

        # # Receive the length of the verification string
        # verify_length_bytes = self.socket.recv(4)
        # verify_length = int.from_bytes(verify_length_bytes, 'little')

        # # Receive the verification string
        # verify_message = self.socket.recv(verify_length).decode('ascii')
        # # print(verify_message)

        return buffer

    def receive_image(self, height, width, num_channels):
        bytes_expected = width * height * num_channels
        buffer = self.receive_bytes(bytes_expected)
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, num_channels)
        return img
    
    def receive_json(self):
        messageLength = self.socket.recv(4)
        messageLength = int.from_bytes(messageLength, 'little')
        message = self.socket.recv(messageLength)
        rec_dict = json.loads(message.decode("utf-8"))

        self.num_timesteps = rec_dict["num_timesteps"]
        self.num_points = rec_dict["num_points"]
        dpg.configure_item("_slider_frame_id", max_value=self.num_timesteps - 1)
        dpg.set_value("_log_num_points", self.num_points)
    
    def reconnect(self):
        try: 
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(1)
            self.socket.connect((self.cfg.host, self.cfg.port))
            print(f"connected to {self.cfg.host}:{self.cfg.port}")
        except Exception as e:
            if self.socket is not None:
                self.socket = None
            print("connection failed, retrying...")
    
    def communicate(self):
        if self.socket is None:
            self.reconnect()
            time.sleep(1)
        else:
            try:
                self.send_json()

                if self.show_splatting or self.show_mesh:
                    img = self.receive_image(self.cam.image_height, self.cam.image_width, 3)
                    self.render_buffer = img.astype(np.float32) / 255.0
                    dpg.set_value("_texture", self.render_buffer)

                self.receive_json()
                self.refresh_fps()
            except Exception as e:
                print("communication interrupted:", e)
                self.socket.close()
                if self.socket is not None:
                    self.socket = None
                time.sleep(1)
    
    def refresh_fps(self):
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{fps:.1f}')
        self.last_time_fresh = time.time()

    def register_dpg(self):

        # register texture =================================================================================================
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        # register window ==================================================================================================
        # the window to display the rendered image
        with dpg.window(label="viewer", tag="_render_window", width=self.W, height=self.H, no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True, no_resize=True):
            dpg.add_image("_texture", width=self.W, height=self.H, tag="_image")

        # control window ==================================================================================================
        with dpg.window(label="Control", tag="_control_window", autosize=True):
            # time
            # if not self.cfg.test:
            #     with dpg.group(horizontal=True):
            #         dpg.add_text("Train time: ")
            #         dpg.add_text("no data", tag="_log_train_time")

            with dpg.group(horizontal=True):
                dpg.add_text("FPS: ")
                dpg.add_text("", tag="_log_fps")

            with dpg.group(horizontal=True):
                dpg.add_text("number of points: ")
                dpg.add_text("", tag="_log_num_points")

            # train button
            with dpg.collapsing_header(label="Train", default_open=True):

                # train / stop
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                            self.elapsed_last += time.time() - self.timestamp_begin
                        else:
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")
                            self.timestamp_begin = time.time()

                    label = 'stop' if self.training else 'start'
                    dpg.add_button(label=label, tag="_button_train", callback=callback_train)

        #                 def callback_reset(sender, app_data):
        #                     @torch.no_grad()
        #                     def weight_reset(m: torch.nn.Module):
        #                         reset_parameters = getattr(m, "reset_parameters", None)
        #                         if callable(reset_parameters):
        #                             m.reset_parameters()

        #                     self.trainer.model.apply(fn=weight_reset)
        #                     self.trainer.model.reset_extra_state()  # for cuda_ray density_grid and step_counter
        #                     self.need_update = True

        #                 dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)

        #             # save ckpt
        #             with dpg.group(horizontal=True):
        #                 dpg.add_text("Checkpoint: ")

        #                 def callback_save(sender, app_data):
        #                     self.trainer.save_checkpoint(full=True, best=False)
        #                     dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
        #                     self.trainer.epoch += 1  # use epoch to indicate different calls.

        #                 dpg.add_button(label="save", tag="_button_save", callback=callback_save)

        #                 dpg.add_text("", tag="_log_ckpt")

        #             # save mesh
        #             with dpg.group(horizontal=True):
        #                 dpg.add_text("Marching Cubes: ")

        #                 def callback_mesh(sender, app_data):
        #                     self.trainer.save_mesh(resolution=256, threshold=10)
        #                     dpg.set_value("_log_mesh", "saved " + f'{self.trainer.name}_{self.trainer.epoch}.ply')
        #                     self.trainer.epoch += 1  # use epoch to indicate different calls.

        #                 dpg.add_button(label="mesh", tag="_button_mesh", callback=callback_mesh)

        #                 dpg.add_text("", tag="_log_mesh")

                    
        #             dpg.add_text("", tag="_log_train_log")
        #             with dpg.collapsing_header(label="losses", default_open=True):
        #                 dpg.add_text("", tag="_log_losses")

        #     # rendering options
            with dpg.collapsing_header(label="Render", default_open=True):

                with dpg.group(horizontal=True):
                    # pause rendering
                    def callback_pause_rendering(sender, app_data):
                        self.pause_rendering = app_data
                        self.need_update = not self.pause_rendering
                    dpg.add_checkbox(label="pause rendering", default_value=self.pause_rendering, callback=callback_pause_rendering)

                    # use original mesh
                    def callback_use_original_mesh(sender, app_data):
                        self.use_original_mesh = app_data
                        self.need_update = True
                    dpg.add_checkbox(label="original mesh", default_value=self.use_original_mesh, callback=callback_use_original_mesh)

        #         # render_mode combo
        #         def callback_change_mode(sender, app_data):
        #             self.render_mode = app_data
        #             self.need_update = True
        #         dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

                with dpg.group(horizontal=True):
                    # show splatting
                    def callback_show_splatting(sender, app_data):
                        self.show_splatting = app_data
                        self.need_update = True
                    dpg.add_checkbox(label="show splatting ", default_value=self.show_splatting, callback=callback_show_splatting)

                    def callback_show_mesh(sender, app_data):
                        self.show_mesh = app_data
                        self.need_update = True
                    dpg.add_checkbox(label="show mesh", default_value=self.canonical_space, callback=callback_show_mesh)

                def callback_set_current_frame(sender, app_data):
                    if sender == "_slider_frame_id":
                        self.timestep = app_data
                    elif sender in ["_button_frame_id_plus", "_mvKey_Right"]:
                        self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                    elif sender in ["_button_frame_id_minus", "_mvKey_Left"]:
                        self.timestep = max(self.timestep - 1, 0)
                    elif sender == "_mvKey_Home":
                        self.timestep = 0
                    elif sender == "_mvKey_End":
                        self.timestep = self.num_timesteps - 1
                    else:
                        print(sender)
                    dpg.set_value("_slider_frame_id", self.timestep)
                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(label="frame id", tag='_slider_frame_id', width=155, min_value=0, max_value=self.num_timesteps - 1, format="%d", default_value=0, callback=callback_set_current_frame)
                    dpg.add_button(label='-', tag="_button_frame_id_minus", callback=callback_set_current_frame)
                    dpg.add_button(label='+', tag="_button_frame_id_plus", callback=callback_set_current_frame)

                # mesh_opacity slider
                def callback_set_opacity(sender, app_data):
                    self.mesh_opacity = app_data
                    if self.show_mesh:
                        self.need_update = True
                dpg.add_slider_float(label="mesh opacity", width=155, min_value=0, max_value=1.0, format="%.2f", default_value=self.mesh_opacity, callback=callback_set_opacity)

        #         # mesh_color picker
        #         def callback_change_mesh_color(sender, app_data):
        #             self.mesh_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
        #             self.need_update = True
        #         dpg.add_color_edit((self.mesh_color*255).tolist(), label="Mesh Color", width=200, no_alpha=True, callback=callback_change_mesh_color)

        #         # bg_color picker
        #         def callback_change_bg(sender, app_data):
        #             self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
        #             self.need_update = True
        #         dpg.add_color_edit((self.bg_color*255).tolist(), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

                # # near slider
                # def callback_set_znear(sender, app_data):
                #     self.cam.znear = app_data
                #     self.need_update = True
                # dpg.add_slider_float(label="near", width=155, min_value=0, max_value=2, format="%.5f", default_value=self.cam.znear, callback=callback_set_znear, tag="_slider_near")

                # # far slider
                # def callback_set_far(sender, app_data):
                #     self.cam.zfar = app_data
                #     self.need_update = True
                # dpg.add_slider_float(label="far", width=155, min_value=1e-3, max_value=2, format="%.5f", default_value=self.cam.zfar, callback=callback_set_far, tag="_slider_far")

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True
                dpg.add_slider_int(label="FoV (vertical)", width=155, min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy")

                # scaling_modifier slider
                def callback_set_scaling_modifier(sender, app_data):
                    self.scaling_modifier = app_data
                    self.need_update = True
                dpg.add_slider_float(label="Scale modifier", width=155, min_value=0, max_value=1, format="%.2f", default_value=self.scaling_modifier, callback=callback_set_scaling_modifier, tag="_slider_scaling_modifier")

                
                dpg.add_separator()
                
                # camera
                with dpg.group(horizontal=True):
                    def callback_reset_camera(sender, app_data):
                        self.cam.reset()
                        self.need_update = True
                        if self.debug:
                            dpg.set_value("_log_pose", str(self.cam.pose.astype(np.float16)))
                        dpg.set_value("_slider_fovy", self.cam.fovy)
                    dpg.add_button(label="reset", tag="_button_reset_pose", callback=callback_reset_camera)
                    with dpg.collapsing_header(label="Camera Pose", default_open=False):
                        dpg.add_text(str(self.cam.pose.astype(np.float16)), tag="_log_pose")
            

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
                dpg.set_value("_slider_frame_id", self.timestep)
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

        dpg.create_viewport(title='GaussianAvatars - Remote Viewer', width=self.W, height=self.H, resizable=True)

        def callback_resize(sender, app_data):
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

        dpg.set_viewport_resize_callback(callback_resize)

        ### global theme ==================================================================================================
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_render_window", theme_no_padding)

        ### finish setup ==================================================================================================
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self):
        while dpg.is_dearpygui_running():
            if self.pause_rendering:
                if self.socket is not None:
                    self.socket.close()
                    self.socket = None
            else:
                self.communicate()
            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    # tyro.cli(main)

    cfg = tyro.cli(Config)
    gui = RemoteViewer(cfg)
    gui.run()
