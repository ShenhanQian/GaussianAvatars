# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#
import os

# # Limit the overall number of Python threads
# max_threads = min(8, os.cpu_count())  # Or set to a specific number like 6
# print(f"Limiting Python threads to {max_threads}")
# # Limit OpenMP threads (used by PyTorch and many scientific libraries)
# os.environ["OMP_NUM_THREADS"] = str(max_threads)
# os.environ["MKL_NUM_THREADS"] = str(max_threads)

import json
import math
import tyro
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path
import time
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib

from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig
from gaussian_renderer import GaussianModel, FlameGaussianModel
from gaussian_renderer import render
from mesh_renderer import NVDiffRenderer

import rclpy
from ros2_blendshape_node import ROS2Subscriber, OVR_ARKIT_BLENDSHAPES_MAP #main as ros2_main
import threading

import logging
import sys
import traceback
import threading
import psutil

from contextlib import contextmanager

import gc


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config(Mini3DViewerConfig):
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    """Pipeline settings for gaussian splatting rendering"""
    cam_convention: Literal["opengl", "opencv"] = "opencv"
    """Camera convention"""
    point_path: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    background_color: tuple[float,float,float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for recording"""
    keyframe_interval: int = 1
    """default keyframe interval"""
    ref_json: Optional[Path] = None
    """ Path to a reference json file. We copy file paths from a reference json into 
    the exported trajectory json file as placeholders so that `render.py` can directly
    load it like a normal sequence. """
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""

ARKit_BLENDSHAPE_NAMES = [
    "browDown_L", "browDown_R", "browInnerUp", "browOuterUp_L", "browOuterUp_R",
    "cheekPuff", "cheekSquint_L", "cheekSquint_R", "eyeBlink_L", "eyeBlink_R",
    "eyeLookDown_L", "eyeLookDown_R", "eyeLookIn_L", "eyeLookIn_R", "eyeLookOut_L",
    "eyeLookOut_R", "eyeLookUp_L", "eyeLookUp_R", "eyeSquint_L", "eyeSquint_R",
    "eyeWide_L", "eyeWide_R", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimple_L", "mouthDimple_R", "mouthFrown_L", "mouthFrown_R",
    "mouthFunnel", "mouthLeft", "mouthLowerDown_L", "mouthLowerDown_R", "mouthPress_L",
    "mouthPress_R", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmile_L", "mouthSmile_R", "mouthStretch_L",
    "mouthStretch_R", "mouthUpperUp_L", "mouthUpperUp_R", "noseSneer_L", "noseSneer_R"
    #"tongueOut"
]

# # Add this near the beginning of your application
# torch.cuda.set_per_process_memory_fraction(0.8)  # Use up to 80% of GPU memory


# # Limit PyTorch threads
# torch.set_num_threads(max_threads)
# torch.set_num_interop_threads(max_threads // 2)  # Typically half the core count


def reserve_gpu_memory(size_mb=2048):
    """Reserve GPU memory at startup to prevent fragmentation"""
    # Convert MB to bytes
    size_bytes = size_mb * 1024 * 1024
    
    # Allocate a large tensor to reserve memory
    # This will be released when the variable goes out of scope
    # but will help establish a larger memory pool
    torch.cuda.empty_cache()
    reserved_memory = torch.empty(size_bytes, device='cuda', dtype=torch.uint8)
    # Now free it to make space for actual operations
    del reserved_memory
    torch.cuda.empty_cache()
    
    logger.info(f"Reserved {size_mb}MB of GPU memory")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('GPUMonitor')

class GPUMonitor:
    def __init__(self, check_interval=1.0):
        self.check_interval = check_interval
        self.keep_running = True
        self.monitor_thread = None
        self.last_allocated = 0
        self.peak_allocated = 0
        self.allocation_spikes = 0
        
    def start(self):
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        self.keep_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        while self.keep_running:
            try:
                # Get PyTorch CUDA memory info with more detail
                if torch.cuda.is_available():
                    try:
                        current_device = torch.cuda.current_device()
                        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2
                        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**2

                        # torch.cuda.empty_cache()
                        
                        # Track memory spikes
                        if memory_allocated > self.peak_allocated:
                            self.peak_allocated = memory_allocated
                            logger.info(f"New peak GPU memory: {self.peak_allocated:.1f}MB")
                        
                        # Log significant changes
                        delta = abs(memory_allocated - self.last_allocated)
                        if delta > 50:  # Log if memory changes by more than 50MB
                            self.allocation_spikes += 1
                            logger.warning(f"Memory spike detected: {delta:.1f}MB change")
                            
                        self.last_allocated = memory_allocated
                        
                        # Log current state
                        logger.info(f"GPU {current_device} Memory: Allocated={memory_allocated:.1f}MB, Reserved={memory_reserved:.1f}MB")

                        torch.cuda.empty_cache()
                        gc.collect()  # Also clear Python garbage collector
                        
                    except Exception as e:
                        logger.warning(f"Could not get GPU memory stats: {e}")
                
                # Log process memory usage
                try:
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    logger.info(f"Process Memory: RSS={memory_info.rss/1024**2:.1f}MB, VMS={memory_info.vms/1024**2:.1f}MB")
                except Exception as e:
                    logger.warning(f"Could not get process memory info: {e}")
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring: {e}")
                
            try:
                time.sleep(self.check_interval)
            except Exception:
                pass

@contextmanager
def cuda_error_handling():
    try:
        yield
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA Runtime Error: {str(e)}")
            logger.error(traceback.format_exc())
            # Try to recover
            torch.cuda.empty_cache()
        else:
            raise
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory: {str(e)}")
        logger.error(traceback.format_exc())
        # Try to recover
        torch.cuda.empty_cache()
        gc.collect()
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        self.setup_error_handling()

        self.threading_lock = threading.RLock() 

        # recording settings
        self.keyframes = []  # list of state dicts of keyframes
        self.all_frames = {}  # state dicts of all frames {key: [num_frames, ...]}
        self.num_record_timeline = 0
        self.playing = False

        print("Initializing 3D Gaussians...")
        self.init_gaussians()


        if self.gaussians.binding is not None:
            # rendering settings
            self.mesh_color = torch.tensor([1, 1, 1, 0.5])
            self.face_colors = None
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)
        
        # FLAME parameters
        if self.gaussians.binding is not None:
            print("Initializing FLAME parameters...")
            self.reset_flame_param()
        
        super().__init__(cfg, 'GaussianAvatars - Local Viewer')

        if self.gaussians.binding is not None:
            self.num_timesteps = self.gaussians.num_timesteps
            dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)

            self.gaussians.select_mesh_by_timestep(self.timestep)

        self.blendshape_values = {name: 0.0 for name in ARKit_BLENDSHAPE_NAMES}
        self.expr_enabled = False
        self.create_blendshape_sliders()

        self.splatting_visible = False

        self.eyes_data = None
        self.avatar_path = None

        self.need_update = False

        rclpy.init()
        self.subscriber = ROS2Subscriber(self)

        # ros2_thread = threading.Thread(target=ros2_main, args=(self,))
        # # # time.sleep(3)
        # ros2_thread.start()

        # # Use a more controlled approach:
        # from concurrent.futures import ThreadPoolExecutor
        
        # # Create a dedicated executor for ROS2
        # self.ros2_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ros2-worker")
        
        # # Submit the ROS2 task to the executor
        # self.ros2_future = self.ros2_executor.submit(ros2_main, self)

        # self.start_random_animation()

    def setup_error_handling(self):
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available - running in CPU mode")
                return

            # Initialize CUDA properly
            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.warning("No CUDA devices found")
                return
                
            # Set device and clear cache
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            
            # Enable gradients tracking for debugging
            torch.set_grad_enabled(True)
            torch.autograd.set_detect_anomaly(True)

            reserve_gpu_memory(2048)
            
            # Start GPU monitoring
            self.gpu_monitor = GPUMonitor(check_interval=10.0)
            self.gpu_monitor.start()
            
            logger.info(f"CUDA setup complete. Using device: {torch.cuda.get_device_name(0)}")
            
        except Exception as e:
            logger.error(f"Error during CUDA setup: {str(e)}")
            logger.error(traceback.format_exc())

    def safe_render(self, cam):
        with cuda_error_handling():
            
            try:
                # Clear cache before render
                torch.cuda.empty_cache()
                
                if dpg.get_value("_checkbox_show_splatting"):
                    # Add sync point to catch errors
                    # torch.cuda.synchronize()
                    with self.threading_lock:
                        rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, 
                                            torch.tensor(self.cfg.background_color).cuda(), 
                                            scaling_modifier=dpg.get_value("_slider_scaling_modifier"))["render"].permute(1, 2, 0).contiguous()
                    
                    # Add another sync point
                    # torch.cuda.synchronize()
                    
                    return rgb_splatting
                    
            except Exception as e:
                logger.error(f"Error in render: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def cleanup(self):
        if hasattr(self, 'gpu_monitor'):
            self.gpu_monitor.stop()
        torch.cuda.empty_cache()

        self.subscriber.destroy_node()
        rclpy.shutdown()

    
        
    def toggle_splatting(self, value=None):
        """
        Toggle or set the splatting visibility
        
        Args:
            value (bool, optional): If provided, set visibility to this value.
                                  If None, toggle current visibility.
        """
        if value is None:
            current_value = dpg.get_value("_checkbox_show_splatting")
            dpg.set_value("_checkbox_show_splatting", not current_value)
        else:
            dpg.set_value("_checkbox_show_splatting", value)
        self.splatting_visible = dpg.get_value("_checkbox_show_splatting")
        print(f"Changed splatting visibilty to {self.splatting_visible}")
        self.need_update = True

    def animate_random_blendshapes(self, duration=5.0, intensity_range=(0.0, 0.7), transition_speed=0.1, 
                               selected_blendshapes=None, random_seed=None):
        """
        Animate random blendshapes over time to create varied facial expressions.
        
        Args:
            duration (float): Duration of the animation in seconds.
            intensity_range (tuple): Range of blendshape intensity values (min, max).
            transition_speed (float): Speed of transitions between expressions (0.0-1.0).
                                    Lower values create smoother transitions.
            selected_blendshapes (list): List of specific blendshapes to animate. If None, uses all blendshapes.
            random_seed (int): Seed for random number generation for reproducible animations.
        
        Note:
            This function should be called in a separate thread to avoid blocking the main UI.
        """
        import numpy as np
        import time
        import threading
        import random
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # If no specific blendshapes selected, use all available ones
        if selected_blendshapes is None:
            available_blendshapes = list(self.blendshape_values.keys())
        else:
            # Ensure all selected blendshapes exist
            available_blendshapes = [bs for bs in selected_blendshapes if bs in self.blendshape_values]
            if not available_blendshapes:
                print("No valid blendshapes selected. Using all available blendshapes.")
                available_blendshapes = list(self.blendshape_values.keys())
        
        # Group related blendshapes to create more natural expressions
        blendshape_groups = {
            'brows': [bs for bs in available_blendshapes if 'brow' in bs.lower()],
            'eyes': [bs for bs in available_blendshapes if 'eye' in bs.lower()],
            'mouth': [bs for bs in available_blendshapes if 'mouth' in bs.lower()],
            'cheeks': [bs for bs in available_blendshapes if 'cheek' in bs.lower()],
            'jaw': [bs for bs in available_blendshapes if 'jaw' in bs.lower()],
            'nose': [bs for bs in available_blendshapes if 'nose' in bs.lower()]
        }
        
        # Create initial and target values for all blendshapes
        current_values = {bs: self.blendshape_values[bs] for bs in available_blendshapes}
        target_values = {bs: 0.0 for bs in available_blendshapes}
        
        # Animation timing variables
        start_time = time.time()
        last_target_update = start_time
        target_update_interval = 1.0/10  # Time between generating new target values
        
        # Clear GPU memory before starting
        if hasattr(self, 'gpu_monitor'):
            torch.cuda.empty_cache()
        
        try:
            with cuda_error_handling():
                # Main animation loop
                while time.time() - start_time < duration:
                    current_time = time.time()
                    
                    # Generate new target values periodically
                    if current_time - last_target_update > target_update_interval:
                        last_target_update = current_time
                        
                        # Reset all targets first
                        for bs in target_values:
                            target_values[bs] = 0.0
                        
                        # Randomly select 1-3 blendshape groups to activate
                        active_groups = random.sample(list(blendshape_groups.keys()), 
                                                    random.randint(1, min(3, len(blendshape_groups))))
                        
                        # For each active group, select a random subset of blendshapes
                        for group in active_groups:
                            group_blendshapes = blendshape_groups[group]
                            if not group_blendshapes:
                                continue
                                
                            # Select 30-70% of blendshapes in this group
                            num_to_select = max(1, int(random.uniform(0.3, 0.7) * len(group_blendshapes)))
                            selected = random.sample(group_blendshapes, num_to_select)
                            
                            # Set random target values for selected blendshapes
                            for bs in selected:
                                min_val, max_val = intensity_range
                                target_values[bs] = random.uniform(min_val, max_val)
                        
                        # Special case: ensure eye blinks are synchronized
                        if 'eyeBlink_L' in target_values and 'eyeBlink_R' in target_values:
                            if random.random() < 0.2:  # 20% chance of blinking
                                blink_value = random.uniform(0.7, 1.0)
                                target_values['eyeBlink_L'] = blink_value
                                target_values['eyeBlink_R'] = blink_value
                    
                    # Interpolate current values toward target values
                    for bs in current_values:
                        current_values[bs] += (target_values[bs] - current_values[bs]) * transition_speed
                        self.blendshape_values[bs] = current_values[bs]
                        
                        # Update UI sliders if they exist
                        try:
                            if dpg.does_item_exist(f"_slider_{bs}"):
                                dpg.set_value(f"_slider_{bs}", current_values[bs])
                        except Exception as e:
                            # Ignore UI update errors
                            pass
                    
                    # Update the FLAME model with new values
                    # self.update_flame_model()
                    self.need_update = True
                    
                    # Allow UI to remain responsive
                    time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error in random blendshapes animation: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Function to start the animation in a separate thread
    def start_random_animation(self, duration=5000.0, intensity=None):
        """Start a random blendshape animation in a background thread"""
        if intensity is None:
            # Choose animation type based on intensity
            animation_thread = threading.Thread(
                target=self.animate_random_blendshapes,
                args=(duration, (0.0, 0.7), 0.1, None, None)
            )
        else:
            animation_thread = threading.Thread(
                target=self.animate_random_blendshapes,
                args=(duration, (0.0, intensity), 0.1, None, None)
            )
        
        animation_thread.daemon = True
        animation_thread.start()
        return animation_thread

    def create_blendshape_sliders(self):
        with dpg.window(label="ARKit Blendshapes", tag="_blendshape_window", autosize=True, pos=(self.W-300, self.H//2),show=not self.cfg.demo_mode):
            # Add reset button
            dpg.add_button(label="Reset Blendshapes", callback=lambda: self.reset_blendshapes())
            
            # # Add sliders grouped by region
            regions = {
                "Brows": ["browDown_L", "browDown_R", "browInnerUp", "browOuterUp_L", "browOuterUp_R"],
                "Eyes": ["eyeBlink_L", "eyeBlink_R", "eyeLookDown_L", "eyeLookDown_R", 
                        "eyeLookIn_L", "eyeLookIn_R", "eyeLookOut_L", "eyeLookOut_R", 
                        "eyeLookUp_L", "eyeLookUp_R", "eyeSquint_L", "eyeSquint_R",
                        "eyeWide_L", "eyeWide_R"],
                "Cheeks": ["cheekPuff", "cheekSquint_L", "cheekSquint_R"],
                "Jaw": ["jawForward", "jawLeft", "jawOpen", "jawRight"],
                "Mouth": ["mouthClose", "mouthDimple_L", "mouthDimple_R", "mouthFrown_L", "mouthFrown_R",
                        "mouthFunnel", "mouthLeft", "mouthLowerDown_L", "mouthLowerDown_R", 
                        "mouthPress_L", "mouthPress_R", "mouthPucker", "mouthRight",
                        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
                        "mouthSmile_L", "mouthSmile_R", "mouthStretch_L", "mouthStretch_R",
                        "mouthUpperUp_L", "mouthUpperUp_R"],
                "Nose": ["noseSneer_L", "noseSneer_R"]
                # "Tongue": ["tongueOut"]
            }

            # regions = {
            #     "Brows": ["Brow Lowerer L", "Brow Lowerer R", "Inner Brow Raiser L", "Outer Brow Raiser L", "Outer Brow Raiser R"],
                
            #     "Eyes": ["Eyes Closed L", "Eyes Closed R", "Eyes Look Down L", "Eyes Look Down R",
            #             "Eyes Look Left L", "Eyes Look Left R", "Eyes Look Right L", "Eyes Look Right R",
            #             "Eyes Look Up L", "Eyes Look Up R", "Lid Tightener L", "Lid Tightener R",
            #             "Upper Lid Raiser L", "Upper Lid Raiser R"],
                        
            #     "Cheeks": ["Cheek Puff L", "Cheek Raiser L", "Cheek Raiser R"],
                
            #     "Jaw": ["Jaw Thrust", "Jaw Sideways Left", "Jaw Drop", "Jaw Sideways Right"],
                
            #     "Mouth": ["Lips Toward", "Dimpler L", "Dimpler R", "Lip Corner Depressor L", "Lip Corner Depressor R",
            #             "Lip Funneler LB", "Mouth Left", "Lower Lip Depressor L", "Lower Lip Depressor R",
            #             "Lip Pressor L", "Lip Pressor R", "Lip Pucker L", "Mouth Right",
            #             "Lip Suck LB", "Lip Suck LT", "Chin Raiser B", "Chin Raiser T",
            #             "Lip Corner Puller L", "Lip Corner Puller R", "Lip Stretcher L", "Lip Stretcher R",
            #             "Upper Lip Raiser L", "Upper Lip Raiser R"],
                        
            #     "Nose": ["Nose Wrinkler L", "Nose Wrinkler R"]
            # }
            
            for region, blendshapes in regions.items():
                with dpg.collapsing_header(label=region, default_open=True):
                    for blendshape in blendshapes:
                        dpg.add_slider_float(
                            label=blendshape,
                            tag=f"_slider_{blendshape}",
                            min_value=0.0,
                            max_value=1.0,
                            default_value=0.0,
                            callback=self.on_blendshape_change,
                            user_data=blendshape,
                            width=250
                        )

    def update_blendshapes_from_ros(self, blendshape_data):
        with self.threading_lock:
            for name, value in blendshape_data.items():
                if name in self.blendshape_values:
                    self.blendshape_values[name] = value
                    try:
                        dpg.set_value(f"_slider_{name}", value)
                    except Exception as e:
                        print(f"Error setting value for {name}: {e}")
            # self.update_flame_model()  # Ensure the FLAME model is updated
            self.need_update = True
            

    def update_eyes_from_ros(self, eyes_data):
        try:
            # Update X values
            dpg.set_value("_slider-eyes-x", eyes_data[0])
            # Manually trigger the callback for X
            # self.flame_param['eyes'][0, 0] = eyes_data[0]  # First eye
            # self.flame_param['eyes'][0, 3] = eyes_data[0]  # Second eye
            
            # Update Y values
            dpg.set_value("_slider-eyes-y", eyes_data[1])
            # Manually trigger the callback for Y
            # self.flame_param['eyes'][0, 1] = eyes_data[1]  # First eye
            # self.flame_param['eyes'][0, 4] = eyes_data[1]  # Second eye
            self.eyes_data = eyes_data
            
            # temporarily take this out to see if confilict with the blendshape causes crash
            # # Update the mesh
            # if not dpg.get_value("_checkbox_enable_control"):
            #     dpg.set_value("_checkbox_enable_control", True)
            # self.gaussians.update_mesh_by_param_dict(self.flame_param)
            self.need_update = True
            
        except Exception as e:
            print(f"Failed to set eye values: {e}")
        


    def update_flame_model(self):
        with self.threading_lock:
            try:
                blendshapes = np.array([self.blendshape_values[name] for name in ARKit_BLENDSHAPE_NAMES])
                expressions, jaw = self.gaussians.flame_model.mask.convert_blendshapes_to_expressions(blendshapes)
                self.flame_param['expr'] = expressions.unsqueeze(0)  # Add batch dimension
                for i in range(len(jaw)): 
                    self.flame_param['jaw'][0, i] = jaw[i]
                    if i==0:
                        self.flame_param['jaw'][0, i] += 0.03 

                self.flame_param['neck'][0, 0] = 0.1 # Neck pitch a bit down for a more natural look

                if self.eyes_data is not None:
                    self.flame_param['eyes'][0, 0] = self.eyes_data[1]  # First eye
                    self.flame_param['eyes'][0, 3] = self.eyes_data[1]  # Second eye
                    self.flame_param['eyes'][0, 1] = -self.eyes_data[0]  # First eye
                    self.flame_param['eyes'][0, 4] = -self.eyes_data[0]  # Second eye

                if not dpg.get_value("_checkbox_enable_control"):
                    dpg.set_value("_checkbox_enable_control", True)


                # torch.cuda.synchronize()
                # torch.cuda.empty_cache()
                self.gaussians.update_mesh_by_param_dict(self.flame_param)
                # torch.cuda.synchronize()
                # self.gaussians.update_mesh_by_param_dict(self.flame_param)


                self.need_update = True
            except Exception as e:
                logger.error(f"Error in blendshape change: {e}")
                logger.error(traceback.format_exc())

    def on_blendshape_change(self, sender, app_data, user_data):
        """Handle changes to blendshape sliders and update FLAME model"""
        blendshape_name = user_data
        blendshape_value = app_data
        
        # Update stored value
        self.blendshape_values[blendshape_name] = blendshape_value

        self.need_update = True

        # self.update_flame_model()
        
        # # Convert blendshapes to array in correct order
        # blendshapes = np.array([self.blendshape_values[name] for name in ARKit_BLENDSHAPE_NAMES])
        
        # # Convert to FLAME expressions
        # expressions = self.gaussians.flame_model.mask.convert_blendshapes_to_expressions(blendshapes)
        
        # # Update FLAME parameters
        # self.flame_param['expr'] = expressions.unsqueeze(0)  # Add batch dimension
        
        # # Enable expression control if not already enabled
        # if not dpg.get_value("_checkbox_enable_control"):
        #     dpg.set_value("_checkbox_enable_control", True)
        
        # # Update mesh with new parameters
        # self.gaussians.update_mesh_by_param_dict(self.flame_param)
        # self.need_update = True

    def reset_blendshapes(self):
        for name in ARKit_BLENDSHAPE_NAMES:
            dpg.set_value(f"_slider_{name}", 0.0)
            self.blendshape_values[name] = 0.0
        self.need_update = True
        # self.update_flame_model()

        # """Reset all blendshape values to 0"""
        # # Reset all sliders in UI
        # for name in ARKit_BLENDSHAPE_NAMES:
        #     dpg.set_value(f"_slider_{name}", 0.0)
        #     self.blendshape_values[name] = 0.0
        
        # # Reset FLAME parameters
        # self.reset_flame_param()
        # if dpg.get_value("_checkbox_enable_control"):
        #     self.gaussians.update_mesh_by_param_dict(self.flame_param)
        # self.need_update = True

    # def create_blendshape_sliders(self):
    #     with dpg.window(label="Blendshapes", tag="_blendshape_window", autosize=True, pos=(self.W-300, self.H//2)):
    #         for blendshape in ARKit_BLENDSHAPE_NAMES:
    #             dpg.add_slider_float(
    #                 label=blendshape,
    #                 min_value=0.0,
    #                 max_value=1.0,
    #                 default_value=0.0,
    #                 callback=self.on_blendshape_change,
    #                 user_data=blendshape,
    #                 width=250
    #             )

    # def on_blendshape_change(self, sender, app_data, user_data):
    #     blendshape_name = user_data
    #     blendshape_value = app_data
    #     # Update the blendshape value in your model
    #     print(f"Blendshape {blendshape_name} changed to {blendshape_value}")
    #     self.need_update = True

    def load_avatar(self, folder_path: Path):
        """
        Load a new avatar and update the viewer state
        
        Args:
            folder_path (Path): Path to the folder containing the avatar files
        """
        if folder_path is None or folder_path == "":
            print("No folder path provided. Cannot load avatar.")
            return
        # Find ply and motion files in the folder
        folder_path = Path(folder_path)
        ply_files = list(folder_path.glob("*.ply"))
        if not ply_files:
            raise FileNotFoundError(f'No .ply file found in {folder_path}')
        point_path = ply_files[0]
        
        motion_path = None
        motion_files = list(folder_path.glob("*.npz"))
        if motion_files:
            motion_path = motion_files[0]
        
        if self.cfg.point_path == point_path and self.cfg.motion_path == motion_path:
            # print("Avatar already loaded.")
            return
        
        self.unload_avatar()
        print("Loading avatar from", folder_path)
        # Update config paths
        self.cfg.point_path = point_path
        self.cfg.motion_path = motion_path
        
        # Clear existing gaussians
        if hasattr(self, 'gaussians'):
            del self.gaussians
            
        # Initialize new gaussians
        if (folder_path / "flame_param.npz").exists():
            self.gaussians = FlameGaussianModel(self.cfg.sh_degree)
        else:
            self.gaussians = GaussianModel(self.cfg.sh_degree)

        # Reset FLAME parameters if applicable
        if self.gaussians.binding is not None:
            self.reset_flame_param()
            
        # Load the new avatar
        self.gaussians.load_ply(point_path, has_target=False, motion_path=motion_path, disable_fid=[])
            
        # Update UI elements
        if self.gaussians.binding is not None:
            self.num_timesteps = self.gaussians.num_timesteps
            dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)
            dpg.set_value("_slider_timestep", 0)
            self.timestep = 0
            self.gaussians.select_mesh_by_timestep(self.timestep)

        # Force update
        self.need_update = True

    def unload_avatar(self):
        """Unload the current avatar and clear related state"""
        if hasattr(self, 'gaussians'):
            del self.gaussians
            self.gaussians = None
            
        self.num_timesteps = 1
        self.timestep = 0
        if dpg.does_item_exist("_slider_timestep"):
            dpg.configure_item("_slider_timestep", max_value=0)
            dpg.set_value("_slider_timestep", 0)
        
        # Clear the render buffer
        self.render_buffer = np.zeros((self.H, self.W, 3))
        if dpg.does_item_exist("_texture"):
            dpg.set_value("_texture", self.render_buffer)
                
        self.need_update = True

    def init_gaussians(self):
        # load gaussians
        if (Path(self.cfg.point_path).parent / "flame_param.npz").exists():
            self.gaussians = FlameGaussianModel(self.cfg.sh_degree)
        else:
            self.gaussians = GaussianModel(self.cfg.sh_degree)

        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['left_half'])
        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['right_half'])
        # unselected_fid = self.gaussians.flame_model.mask.get_fid_except_fids(selected_fid)
        unselected_fid = []
        
        if self.cfg.point_path is not None:
            if self.cfg.point_path.exists():
                self.gaussians.load_ply(self.cfg.point_path, has_target=False, motion_path=self.cfg.motion_path, disable_fid=unselected_fid)
            else:
                raise FileNotFoundError(f'{self.cfg.point_path} does not exist.')

    def refresh_stat(self):
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{int(fps):<4d}')
        self.last_time_fresh = time.time()
    
    def update_record_timeline(self):
        cycles = dpg.get_value("_input_cycles")
        if cycles == 0:
            self.num_record_timeline = sum([keyframe['interval'] for keyframe in self.keyframes[:-1]])
        else:
            self.num_record_timeline = sum([keyframe['interval'] for keyframe in self.keyframes]) * cycles

        dpg.configure_item("_slider_record_timestep", min_value=0, max_value=self.num_record_timeline-1)

        if len(self.keyframes) <= 0:
            self.all_frames = {}
            return
        else:
            k_x = []

            keyframes = self.keyframes.copy()
            if cycles > 0:
                # pad a cycle at the beginning and the end to ensure smooth transition
                keyframes = self.keyframes * (cycles + 2)
                t_couter = -sum([keyframe['interval'] for keyframe in self.keyframes])
            else:
                t_couter = 0

            for keyframe in keyframes:
                k_x.append(t_couter)
                t_couter += keyframe['interval']
            
            x = np.arange(self.num_record_timeline)
            self.all_frames = {}

            if len(keyframes) <= 1:
                for k in keyframes[0]:
                    k_y = np.concatenate([np.array(keyframe[k])[None] for keyframe in keyframes], axis=0)
                    self.all_frames[k] = np.tile(k_y, (self.num_record_timeline, 1))
            else:
                kind = 'linear' if len(keyframes) <= 3 else 'cubic'
            
                for k in keyframes[0]:
                    if k == 'interval':
                        continue
                    k_y = np.concatenate([np.array(keyframe[k])[None] for keyframe in keyframes], axis=0)
                  
                    interp_funcs = [interp1d(k_x, k_y[:, i], kind=kind, fill_value='extrapolate') for i in range(k_y.shape[1])]

                    y = np.array([interp_func(x) for interp_func in interp_funcs]).transpose(1, 0)
                    self.all_frames[k] = y

    def get_state_dict(self):
        return {
            'rot': self.cam.rot.as_quat(),
            'look_at': np.array(self.cam.look_at),
            'radius': np.array([self.cam.radius]).astype(np.float32),
            'fovy': np.array([self.cam.fovy]).astype(np.float32),
            'interval': self.cfg.fps*self.cfg.keyframe_interval,
        }

    def get_state_dict_record(self):
        record_timestep = dpg.get_value("_slider_record_timestep")
        state_dict = {k: self.all_frames[k][record_timestep] for k in self.all_frames}
        return state_dict

    def apply_state_dict(self, state_dict):
        if 'rot' in state_dict:
            self.cam.rot = R.from_quat(state_dict['rot'])
        if 'look_at' in state_dict:
            self.cam.look_at = state_dict['look_at']
        if 'radius' in state_dict:
            self.cam.radius = state_dict['radius'].item()
        if 'fovy' in state_dict:
            self.cam.fovy = state_dict['fovy'].item()
    
    def parse_ref_json(self):
        if self.cfg.ref_json is None:
            return {}
        else:
            with open(self.cfg.ref_json, 'r') as f:
                ref_dict = json.load(f)

        tid2paths = {}
        for frame in ref_dict['frames']:
            tid = frame['timestep_index']
            if tid not in tid2paths:
                tid2paths[tid] = frame
        return tid2paths
    
    def export_trajectory(self):
        tid2paths = self.parse_ref_json()

        if self.num_record_timeline <= 0:
            return
        
        timestamp = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        traj_dict = {'frames': []}
        timestep_indices = []
        camera_indices = []
        for i in range(self.num_record_timeline):
            # update
            dpg.set_value("_slider_record_timestep", i)
            state_dict = self.get_state_dict_record()
            self.apply_state_dict(state_dict)

            self.need_update = True
            while self.need_update:
                time.sleep(0.001)

            # save image
            save_folder = self.cfg.save_folder / timestamp
            if not save_folder.exists():
                save_folder.mkdir(parents=True)
            path = save_folder / f"{i:05d}.png"
            print(f"Saving image to {path}")
            Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)

            # cache camera parameters
            cx = self.cam.intrinsics[2]
            cy = self.cam.intrinsics[3]
            fl_x = self.cam.intrinsics[0].item() if isinstance(self.cam.intrinsics[0], np.ndarray) else self.cam.intrinsics[0]
            fl_y = self.cam.intrinsics[1].item() if isinstance(self.cam.intrinsics[1], np.ndarray) else self.cam.intrinsics[1]
            h = self.cam.image_height
            w = self.cam.image_width
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2

            c2w = self.cam.pose.copy()  # opencv convention
            c2w[:, [1, 2]] *= -1  # opencv to opengl
            # transform_matrix = np.linalg.inv(c2w).tolist()  # world2cam
            
            timestep_index = self.timestep
            camera_indx = i
            timestep_indices.append(timestep_index)
            camera_indices.append(camera_indx)
            
            tid2paths[timestep_index]['file_path']

            frame = {
                "cx": cx,
                "cy": cy,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "h": h,
                "w": w,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
                "transform_matrix": c2w.tolist(),
                'timestep_index': timestep_index,
                'camera_indx': camera_indx,
            }
            if timestep_index in tid2paths:
                frame['file_path'] = tid2paths[timestep_index]['file_path']
                frame['fg_mask_path'] = tid2paths[timestep_index]['fg_mask_path']
                frame['flame_param_path'] = tid2paths[timestep_index]['flame_param_path']
            traj_dict['frames'].append(frame)

            # update timestep
            if dpg.get_value("_checkbox_dynamic_record"):
                self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                dpg.set_value("_slider_timestep", self.timestep)
                self.gaussians.select_mesh_by_timestep(self.timestep)
        
        traj_dict['timestep_indices'] = sorted(list(set(timestep_indices)))
        traj_dict['camera_indices'] = sorted(list(set(camera_indices)))
        
        # save camera parameters
        path = save_folder / f"trajectory.json"
        print(f"Saving trajectory to {path}")
        with open(path, 'w') as f:
            json.dump(traj_dict, f, indent=4)

    def reset_flame_param(self):
        self.flame_param = {
            'expr': torch.zeros(1, self.gaussians.n_expr),
            'rotation': torch.zeros(1, 3),
            'neck': torch.zeros(1, 3),
            'jaw': torch.zeros(1, 3),
            'eyes': torch.zeros(1, 6),
            'translation': torch.zeros(1, 3),
        }

    def define_gui(self):
        super().define_gui()

        # window: rendering options ==================================================================================================
        with dpg.window(label="Render", tag="_render_window", autosize=True,  show=not self.cfg.demo_mode):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS:", show=not self.cfg.demo_mode)
                dpg.add_text("0   ", tag="_log_fps", show=not self.cfg.demo_mode)

            dpg.add_text(f"number of points: {self.gaussians._xyz.shape[0]}")
            
            with dpg.group(horizontal=True):
                # show splatting
                def callback_show_splatting(sender, app_data):
                    self.need_update = True
                dpg.add_checkbox(label="show splatting", default_value=False, callback=callback_show_splatting, tag="_checkbox_show_splatting")
                self.splatting_visible = dpg.get_value("_checkbox_show_splatting")

                dpg.add_spacer(width=10)

                if self.gaussians.binding is not None:
                    # show mesh
                    def callback_show_mesh(sender, app_data):
                        self.need_update = True
                    dpg.add_checkbox(label="show mesh", default_value=False, callback=callback_show_mesh, tag="_checkbox_show_mesh")

                    # # show original mesh
                    # def callback_original_mesh(sender, app_data):
                    #     self.original_mesh = app_data
                    #     self.need_update = True
                    # dpg.add_checkbox(label="original mesh", default_value=self.original_mesh, callback=callback_original_mesh)
            
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
                    dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=153, min_value=0, max_value=self.num_timesteps - 1, format="%d", default_value=0, callback=callback_set_current_frame)

            # # render_mode combo
            # def callback_change_mode(sender, app_data):
            #     self.render_mode = app_data
            #     self.need_update = True
            # dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

            # scaling_modifier slider
            def callback_set_scaling_modifier(sender, app_data):
                self.need_update = True
            dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=2, format="%.2f", width=200, default_value=1.0, callback=callback_set_scaling_modifier, tag="_slider_scaling_modifier")

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True
            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, width=200, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy", show=not self.cfg.demo_mode)

            if self.gaussians.binding is not None:
                # visualization options
                def callback_visual_options(sender, app_data):
                    if app_data == 'number of points per face':
                        value, ct = self.gaussians.binding.unique(return_counts=True)
                        ct = torch.log10(ct + 1)
                        ct = ct.float() / ct.max()
                        cmap = matplotlib.colormaps["plasma"]
                        self.face_colors = torch.from_numpy(cmap(ct.cpu())[None, :, :3]).to(self.gaussians.verts)
                    else:
                        self.face_colors = self.mesh_color[:3].to(self.gaussians.verts)[None, None, :].repeat(1, self.gaussians.face_center.shape[0], 1)  # (1, F, 3)
                    
                    dpg.set_value('_checkbox_show_mesh', True)
                    self.need_update = True
                dpg.add_combo(["none", "number of points per face"], default_value="none", label='visualization', width=200, callback=callback_visual_options, tag="_visual_options")

                # mesh_color picker
                def callback_change_mesh_color(sender, app_data):
                    self.mesh_color = torch.tensor(app_data, dtype=torch.float32)  # only need RGB in [0, 1]
                    if dpg.get_value("_visual_options") == 'none':
                        self.face_colors = self.mesh_color[:3].to(self.gaussians.verts)[None, None, :].repeat(1, self.gaussians.face_center.shape[0], 1)
                    self.need_update = True
                dpg.add_color_edit((self.mesh_color*255).tolist(), label="Mesh Color", width=200, callback=callback_change_mesh_color, show=not self.cfg.demo_mode)

            # # bg_color picker
            # def callback_change_bg(sender, app_data):
            #     self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
            #     self.need_update = True
            # dpg.add_color_edit((self.bg_color*255).tolist(), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

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
            
            # camera
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                    dpg.set_value("_slider_fovy", self.cam.fovy)
                dpg.add_button(label="reset camera", tag="_button_reset_pose", callback=callback_reset_camera, show=not self.cfg.demo_mode)
                
                def callback_cache_camera(sender, app_data):
                    self.cam.save()
                dpg.add_button(label="cache camera", tag="_button_cache_pose", callback=callback_cache_camera, show=not self.cfg.demo_mode)

                def callback_clear_cache(sender, app_data):
                    self.cam.clear()
                dpg.add_button(label="clear cache", tag="_button_clear_cache", callback=callback_clear_cache, show=not self.cfg.demo_mode)
                
        # window: recording ==================================================================================================
        with dpg.window(label="Record", tag="_record_window", autosize=True, pos=(0, self.H//2), show=not self.cfg.demo_mode):
            dpg.add_text("Keyframes")
            with dpg.group(horizontal=True):
                # list keyframes
                def callback_set_current_keyframe(sender, app_data):
                    idx = int(dpg.get_value("_listbox_keyframes"))
                    self.apply_state_dict(self.keyframes[idx])

                    record_timestep = sum([keyframe['interval'] for keyframe in self.keyframes[:idx]])
                    dpg.set_value("_slider_record_timestep", record_timestep)

                    self.need_update = True
                dpg.add_listbox(self.keyframes, width=200, tag="_listbox_keyframes", callback=callback_set_current_keyframe)

                # edit keyframes
                with dpg.group():
                    # add
                    def callback_add_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            new_idx = 0
                        else:
                            new_idx = int(dpg.get_value("_listbox_keyframes")) + 1

                        states = self.get_state_dict()
                        
                        self.keyframes.insert(new_idx, states)
                        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", new_idx)

                        self.update_record_timeline()
                    dpg.add_button(label="add", tag="_button_add_keyframe", callback=callback_add_keyframe)

                    # delete
                    def callback_delete_keyframe(sender, app_data):
                        idx = int(dpg.get_value("_listbox_keyframes"))
                        self.keyframes.pop(idx)
                        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", idx-1)

                        self.update_record_timeline()
                    dpg.add_button(label="delete", tag="_button_delete_keyframe", callback=callback_delete_keyframe)

                    # update
                    def callback_update_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            return
                        else:
                            idx = int(dpg.get_value("_listbox_keyframes"))

                        states = self.get_state_dict()
                        states['interval'] = self.cfg.fps*self.cfg.keyframe_interval

                        self.keyframes[idx] = states
                    dpg.add_button(label="update", tag="_button_update_keyframe", callback=callback_update_keyframe)

            with dpg.group(horizontal=True):
                def callback_set_record_cycles(sender, app_data):
                    self.update_record_timeline()
                dpg.add_input_int(label="cycles", tag="_input_cycles", default_value=0, width=70, callback=callback_set_record_cycles)

                def callback_set_keyframe_interval(sender, app_data):
                    self.cfg.keyframe_interval = app_data
                    for keyframe in self.keyframes:
                        keyframe['interval'] = self.cfg.fps*self.cfg.keyframe_interval
                    self.update_record_timeline()
                dpg.add_input_int(label="interval", tag="_input_interval", default_value=self.cfg.keyframe_interval, width=70, callback=callback_set_keyframe_interval)
            
            def callback_set_record_timestep(sender, app_data):
                state_dict = self.get_state_dict_record()
                
                self.apply_state_dict(state_dict)
                self.need_update = True
            dpg.add_slider_int(label="timeline", tag='_slider_record_timestep', width=200, min_value=0, max_value=0, format="%d", default_value=0, callback=callback_set_record_timestep)
            
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="dynamic", default_value=False, tag="_checkbox_dynamic_record")
                dpg.add_checkbox(label="loop", default_value=True, tag="_checkbox_loop_record")
            
            with dpg.group(horizontal=True):
                def callback_play(sender, app_data):
                    self.playing = not self.playing
                    self.need_update = True
                dpg.add_button(label="play", tag="_button_play", callback=callback_play)

                def callback_export_trajectory(sender, app_data):
                    self.export_trajectory()
                dpg.add_button(label="export traj", tag="_button_export_traj", callback=callback_export_trajectory)
            
            def callback_save_image(sender, app_data):
                if not self.cfg.save_folder.exists():
                    self.cfg.save_folder.mkdir(parents=True)
                path = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                print(f"Saving image to {path}")
                Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
            with dpg.group(horizontal=True):
                dpg.add_button(label="save image", tag="_button_save_image", callback=callback_save_image)

        # window: FLAME ==================================================================================================
        if self.gaussians.binding is not None:
            with dpg.window(label="FLAME parameters", tag="_flame_window", autosize=True, pos=(self.W-300, 0),show=not self.cfg.demo_mode):
                def callback_enable_control(sender, app_data):
                    if app_data:
                        self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    else:
                        self.gaussians.select_mesh_by_timestep(self.timestep)
                    self.need_update = True
                dpg.add_checkbox(label="enable control", default_value=False, tag="_checkbox_enable_control", callback=callback_enable_control)

                dpg.add_separator()

                def callback_set_pose(sender, app_data):
                    
                    joint, axis = sender.split('-')[1:3]
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    self.flame_param[joint][0, axis_idx] = app_data
                    if joint == 'eyes':
                        print("in the callback to set eye pose")
                        self.flame_param[joint][0, 3+axis_idx] = app_data
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                dpg.add_text(f'Joints')
                self.pose_sliders = []
                max_rot = 0.5
                for joint in ['neck', 'jaw', 'eyes']:
                    if joint in self.flame_param:
                        with dpg.group(horizontal=True):
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 0], callback=callback_set_pose, tag=f"_slider-{joint}-x", width=70)
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 1], callback=callback_set_pose, tag=f"_slider-{joint}-y", width=70)
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 2], callback=callback_set_pose, tag=f"_slider-{joint}-z", width=70)
                            self.pose_sliders.append(f"_slider-{joint}-x")
                            self.pose_sliders.append(f"_slider-{joint}-y")
                            self.pose_sliders.append(f"_slider-{joint}-z")
                            dpg.add_text(f'{joint:4s}')
                dpg.add_text('   roll       pitch      yaw')
                
                dpg.add_separator()
                
                def callback_set_expr(sender, app_data):
                    expr_i = int(sender.split('-')[2])
                    self.flame_param['expr'][0, expr_i] = app_data
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                self.expr_sliders = []
                dpg.add_text(f'Expressions')
                for i in range(5):
                    dpg.add_slider_float(label=f"{i}", min_value=-3, max_value=3, format="%.2f", default_value=0, callback=callback_set_expr, tag=f"_slider-expr-{i}", width=250)
                    self.expr_sliders.append(f"_slider-expr-{i}")

                def callback_reset_flame(sender, app_data):
                    self.reset_flame_param()
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                    for slider in self.pose_sliders + self.expr_sliders:
                        dpg.set_value(slider, 0)
                dpg.add_button(label="reset FLAME", tag="_button_reset_flame", callback=callback_reset_flame)

        # widget-dependent handlers ========================================================================================
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

            def callbackmouse_wheel_slider(sender, app_data):
                delta = app_data
                if dpg.is_item_hovered("_slider_timestep"):
                    self.timestep = min(max(self.timestep - delta, 0), self.num_timesteps - 1)
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)
                    self.need_update = True
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel_slider)

            def callback_toggle_splatting(sender, app_data):
                self.toggle_splatting()
            dpg.add_key_press_handler(dpg.mvKey_S, callback=callback_toggle_splatting)

            def callback_load_alona(sender, app_data):
                self.unload_avatar()
                # To load a new avatar:
                self.load_avatar(Path("/workspace/avatars/alona5/"))

            def callback_load_tatjana(sender, app_data):
                self.unload_avatar()
                # To load a new avatar:
                self.load_avatar(Path("/workspace/avatars/tatjana2/"))

            dpg.add_key_press_handler(dpg.mvKey_A, callback=callback_load_alona) 
            dpg.add_key_press_handler(dpg.mvKey_T, callback=callback_load_tatjana)   


    
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
        try:
            print("Running LocalViewer...")
            target_fps = 30
            frame_time = 1.0 / target_fps  # Time per frame in seconds
            last_frame_time = time.time()

            while dpg.is_dearpygui_running():
                try:
                    

                    current_time = time.time()
                    elapsed = current_time - last_frame_time

                    if rclpy.ok():
                        rclpy.spin_once(self.subscriber, timeout_sec=0)
                        if self.subscriber.eyes_data is not None:
                            self.update_eyes_from_ros(self.subscriber.eyes_data)
                        if self.subscriber.blendshape_data is not None:
                            self.update_blendshapes_from_ros(self.subscriber.blendshape_data)
                        splatting_visible_cur = dpg.get_value("_checkbox_show_splatting")
                        if splatting_visible_cur != self.splatting_visible:
                            self.toggle_splatting(self.splatting_visible)
                        if self.avatar_path != self.cfg.point_path and self.avatar_path is not None and self.avatar_path != "":
                            self.load_avatar(self.avatar_path)
                        
              

                    if self.need_update or self.playing:
                        if not hasattr(self, 'gaussians') or self.gaussians is None:
                            self.render_buffer = np.zeros((self.H, self.W, 3))
                            dpg.set_value("_texture", self.render_buffer)
                            self.need_update = False
                            continue

                        self.update_flame_model()  # Ensure the FLAME model is updated

                        cam = self.prepare_camera()
                        # time.sleep(0.05)

                        if dpg.get_value("_checkbox_show_splatting"):
                            with cuda_error_handling():
                                # rgb_splatting = self.safe_render(cam)
                                # rgb
                                # torch.cuda.synchronize()
                                rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, torch.tensor(self.cfg.background_color).cuda(), scaling_modifier=dpg.get_value("_slider_scaling_modifier"))["render"].permute(1, 2, 0).contiguous()
                                # torch.cuda.synchronize()
                                # opacity
                                # override_color = torch.ones_like(self.gaussians._xyz).cuda()
                                # background_color = torch.tensor(self.cfg.background_color).cuda() * 0
                                # rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, background_color, scaling_modifier=dpg.get_value("_slider_scaling_modifier"), override_color=override_color)["render"].permute(1, 2, 0).contiguous()

                        if self.gaussians.binding is not None and dpg.get_value("_checkbox_show_mesh"):
                            out_dict = self.mesh_renderer.render_from_camera(self.gaussians.verts, self.gaussians.faces, cam, face_colors=self.face_colors)

                            rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                            rgb_mesh = rgba_mesh[:, :, :3]
                            alpha_mesh = rgba_mesh[:, :, 3:]
                            mesh_opacity = self.mesh_color[3:].cuda()

                        if dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                            rgb = rgb_mesh * alpha_mesh * mesh_opacity  + rgb_splatting * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                        elif dpg.get_value("_checkbox_show_splatting") and not dpg.get_value("_checkbox_show_mesh"):
                            rgb = rgb_splatting
                        elif not dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                            rgb = rgb_mesh
                        else:
                            rgb = torch.zeros([self.H, self.W, 3])

                        self.render_buffer = rgb.cpu().numpy()
                        if self.render_buffer.shape[0] != self.H or self.render_buffer.shape[1] != self.W:
                            continue
                        dpg.set_value("_texture", self.render_buffer)

                        self.refresh_stat()
                        self.need_update = False

                        if self.playing:
                            record_timestep = dpg.get_value("_slider_record_timestep")
                            if record_timestep >= self.num_record_timeline - 1:
                                if not dpg.get_value("_checkbox_loop_record"):
                                    self.playing = False
                                dpg.set_value("_slider_record_timestep", 0)
                            else:
                                dpg.set_value("_slider_record_timestep", record_timestep + 1)
                                if dpg.get_value("_checkbox_dynamic_record"):
                                    self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                                    dpg.set_value("_slider_timestep", self.timestep)
                                    self.gaussians.select_mesh_by_timestep(self.timestep)

                                state_dict = self.get_state_dict_record()
                                self.apply_state_dict(state_dict)
                        
                    
                    frame_duration = time.time() - current_time
                    sleep_time = max(0, frame_time - frame_duration)
                    # if sleep_time > 0:
                    #     time.sleep(sleep_time)
                    
                    # Update last frame time for next iteration
                    last_frame_time = time.time()

                except Exception as e:
                    logger.error(f"Error in render loop: {str(e)}")
                    logger.error(traceback.format_exc())

                dpg.render_dearpygui_frame()
        finally:
            self.cleanup()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = LocalViewer(cfg)
    gui.run()

