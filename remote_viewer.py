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

from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig


@dataclass
class Config(Mini3DViewerConfig):
    cam_convention: Literal["opengl", "opencv"] = "opengl"
    """Camera convention"""
    host: str = "127.0.0.1"
    port: int = 6009
    pause_rendering: bool = False
    training: bool = True
    """start training at launch time"""

class RemoteViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # training settings
        self.training = cfg.training
        self.pause_rendering = cfg.pause_rendering
        self.step = 0  # training step
        self.timestamp_begin = time.time() if self.training else None
        self.elapsed_last = 0  # training time

        # network
        self.socket = None

        super().__init__(cfg, 'GaussianAvatars - Remote Viewer')

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
                "scaling_modifier": dpg.get_value("_slider_scaling_modifier"),
                "show_splatting": dpg.get_value("_checkbox_show_splatting"),
                "show_mesh": dpg.get_value("_checkbox_show_mesh"),
                "mesh_opacity": dpg.get_value("_slider_mesh_opacity"),
                "use_original_mesh": dpg.get_value("_checkbox_use_original_mesh"),
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
        dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)
        dpg.set_value("_log_num_points", rec_dict["num_points"])
    
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

                if dpg.get_value('_checkbox_show_splatting') or dpg.get_value('_checkbox_show_mesh'):
                    img = self.receive_image(self.cam.image_height, self.cam.image_width, 3)
                    self.render_buffer = img.astype(np.float32) / 255.0
                    if img.shape[0] == self.cam.image_height and img.shape[1] == self.cam.image_width:
                        dpg.set_value("_texture", self.render_buffer)

                self.receive_json()
                self.refresh_stat()
            except Exception as e:
                print("communication interrupted:", e)
                self.socket.close()
                if self.socket is not None:
                    self.socket = None
                time.sleep(1)
    
    def refresh_stat(self):
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{fps:.1f}')
        self.last_time_fresh = time.time()

        dpg.set_value("_log_pose", str(self.cam.pose.astype(np.float16)))

    def define_gui(self):
        super().define_gui()

        # control window ==================================================================================================
        with dpg.window(label="Control", tag="_control_window", autosize=True):
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

            # rendering options
            with dpg.collapsing_header(label="Render", default_open=True):

                with dpg.group(horizontal=True):
                    # pause rendering
                    def callback_pause_rendering(sender, app_data):
                        self.pause_rendering = app_data
                        self.need_update = not self.pause_rendering
                    dpg.add_checkbox(label="pause rendering", default_value=self.pause_rendering, callback=callback_pause_rendering)

                    # use original mesh
                    def callback_use_original_mesh(sender, app_data):
                        self.need_update = True
                    dpg.add_checkbox(label="original mesh", default_value=False, callback=callback_use_original_mesh, tag="_checkbox_use_original_mesh")

                # # render_mode combo
                # def callback_change_mode(sender, app_data):
                #     self.render_mode = app_data
                #     self.need_update = True
                # dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

                with dpg.group(horizontal=True):
                    # show splatting
                    def callback_show_splatting(sender, app_data):
                        self.need_update = True
                    dpg.add_checkbox(label="show splatting ", default_value=True, callback=callback_show_splatting, tag="_checkbox_show_splatting")

                    def callback_show_mesh(sender, app_data):
                        self.need_update = True
                    dpg.add_checkbox(label="show mesh", default_value=False, callback=callback_show_mesh, tag="_checkbox_show_mesh")

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
                    else:
                        print(sender)
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=155, min_value=0, max_value=self.num_timesteps - 1, format="%d", default_value=0, callback=callback_set_current_frame)
                    dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                    dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)

                # mesh_opacity slider
                def callback_set_opacity(sender, app_data):
                    if dpg.get_value("_checkbox_show_mesh"):
                        self.need_update = True
                dpg.add_slider_float(label="mesh opacity", width=155, min_value=0, max_value=1.0, format="%.2f", default_value=0.5, callback=callback_set_opacity, tag="_slider_mesh_opacity")

                # # mesh_color picker
                # def callback_change_mesh_color(sender, app_data):
                #     self.mesh_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                #     self.need_update = True
                # dpg.add_color_edit((self.mesh_color*255).tolist(), label="Mesh Color", width=200, no_alpha=True, callback=callback_change_mesh_color)

                # # bg_color picker
                # def callback_change_bg(sender, app_data):
                #     self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                #     self.need_update = True
                # dpg.add_color_edit((self.bg_color*255).tolist(), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

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
                    self.need_update = True
                dpg.add_slider_float(label="Scale modifier", width=155, min_value=0, max_value=1, format="%.2f", default_value=1, callback=callback_set_scaling_modifier, tag="_slider_scaling_modifier")

                
                dpg.add_separator()
                
                # camera
                with dpg.group(horizontal=True):
                    def callback_reset_camera(sender, app_data):
                        self.cam.reset()
                        self.need_update = True
                        dpg.set_value("_log_pose", str(self.cam.pose.astype(np.float16)))
                        dpg.set_value("_slider_fovy", self.cam.fovy)
                    dpg.add_button(label="reset", tag="_button_reset_pose", callback=callback_reset_camera)
                    with dpg.collapsing_header(label="Camera Pose", default_open=False):
                        dpg.add_text(str(self.cam.pose.astype(np.float16)), tag="_log_pose")
                
        # widget-dependent handlers ========================================================================================
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

            def callbackmouse_wheel(sender, app_data):
                delta = app_data
                if dpg.is_item_hovered("_slider_timestep"):
                    self.timestep = min(max(self.timestep - delta, 0), self.num_timesteps - 1)
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.need_update = True
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel)

            
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
    cfg = tyro.cli(Config)
    gui = RemoteViewer(cfg)
    gui.run()
