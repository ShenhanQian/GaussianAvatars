# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple
import tyro


class Config:
    def __getitem__(self, __name: str):
        if hasattr(self, __name):
            return getattr(self, __name)
        else:
            return None


@dataclass()
class DataConfig(Config):
    root_folder: Path = ""
    """The root folder for the dataset."""
    division: Optional[str] = None
    subset: Optional[str] = None
    calibrated: bool = True
    align_cameras_to_axes: bool = True
    """Adjust how cameras distribute in the space with a global rotation"""
    camera_coord_conversion: str = 'opencv->opengl'
    n_downsample_rgb: Optional[int] = None
    """Load from downsampled RGB images to save data IO time"""
    scale_factor: float = 1.0
    """Further apply a scaling transformation after the downsampling of RGB"""
    background_color: Optional[Literal['white', 'black']] = None
    use_alpha_map: bool = False
    use_landmark: bool = True


@dataclass()
class ModelConfig(Config):
    n_shape: int = 300
    n_expr: int = 100
    n_tex: int = 100

    use_static_offset: bool = True
    """Optimize static offsets on top of FLAME vertices in the canonical space"""
    use_dynamic_offset: bool = False
    """Optimize dynamic offsets on top of the FLAME vertices in the canonical space"""
    add_teeth: bool = True
    """Add teeth to the FLAME model"""

    tex_resolution: int = 2048
    """The resolution of the extra texture map"""
    tex_painted: bool = True
    """Use a painted texture map instead the pca texture space as the base texture map"""
    tex_extra: bool = True
    """Optimize an extra texture map as the base texture map or the residual texture map"""
    # tex_clusters: tuple[str, ...] = ("skin", "hair", "sclerae", "lips_tight", "boundary")
    tex_clusters: tuple[str, ...] = ("skin", "hair", "boundary", "lips_tight", "teeth", "sclerae", "irises")
    """Regions that are supposed to share a similar color inside"""
    residual_tex: bool = True
    """Use the extra texture map as a residual component on top of the base texture"""
    occluded: tuple[str, ...] = ()  # to be used for updating stage configs in __post_init__
    """The regions that are occluded by the hair or garments"""
    
    flame_params_path: Optional[Path] = None


@dataclass()
class RenderConfig(Config):
    """Rendering configurations"""
    sampling_scale: float = 2
    backend: Literal['nvdiffrast', 'pytorch3d'] = 'nvdiffrast'
    diff_rast: bool = True
    """Require gradients from the rasterizer"""
    use_opengl: bool = True
    """Use OpenGL for NVDiffRast"""
    background_train: Literal['white', 'black', 'target'] = 'target'
    """Background color/image for training"""
    background_eval: Literal['white', 'black', 'target'] = 'target'
    """Background color/image for evaluation"""
    blur_sigma: Tuple[float, float] = (10, 1e-4)
    """The initial and the final values for the blur coefficient sigma"""
    lighting_type: Literal['constant', 'front', 'front-range', 'SH'] = 'SH'
    """The type of lighting"""
    lighting_space: Literal['world', 'camera'] = 'world'
    """The space of lighting"""


@dataclass()
class LearningRateConfig(Config):
    """Learning rates configurations"""
    base: float = 5e-3
    """shape, texture, rotation, eyes, neck, jaw"""
    translation: float = 1e-3
    expr: float = 5e-2  # NOTE: tuned
    static_offset: float = 5e-4  # NOTE: tuned
    dynamic_offset: float = 5e-4  # NOTE: tuned
    camera: float = 5e-3
    light: float = 5e-3


@dataclass()
class LossWeightConfig(Config):
    """Loss weights configurations"""
    landmark: Optional[float] = 3.  # NOTE: tuned (should not be lower to avoid collapse)
    disable_boundary_landmarks: bool = True
    """Disable the landmark loss for the boundary landmarks since they are not accurate"""

    photo: Optional[float] = 30.

    reg_shape: float = 3e-1  # NOTE: tuned
    reg_expr: float = 1e-2  # NOTE: tuned (for best expressivness)
    reg_tex_pca: float = 1e-4  # NOTE: tuned (will make it hard to model hair color when too high)
    
    reg_tex_res: Optional[float] = None  # 1e2 (when w/o reg_var)  # NOTE: tuned
    """Regularize the residual texture map"""
    reg_tex_res_clusters: Optional[float] = 1e1  # NOTE: tuned
    """Regularize the residual texture map inside each texture cluster"""
    reg_tex_res_for: tuple[str, ...] = ("sclerae", "teeth")
    reg_tex_var: Optional[float] = None  # 1e2
    """Regularize the variance of the texture map inside each texture cluster"""
    reg_tex_tv: Optional[float] = 1e5  # NOTE: tuned  (important to split regions apart)
    """Regularize the total variation of the texture map"""

    reg_light: Optional[float] = None  # NOTE: tuned
    """Regularize lighting parameters"""
    reg_diffuse: Optional[float] = 1e2  # NOTE: tuned
    """Regularize lighting parameters by the diffuse term"""

    reg_offset: Optional[float] = 3e1  #1e4  # NOTE: tuned
    """Regularize the norm of offsets"""
    reg_offset_relax_coef: float = 1.  #0.01  # NOTE: tuned
    """The coefficient for relaxing reg_offset for the regions specified"""
    reg_offset_relax_for: tuple[str, ...] = ("hair", "ears")
    """Relax the offset loss for the regions specified"""

    reg_offset_lap: Optional[float] = 1e6  #3e6  # NOTE: tuned
    """Regularize the difference of laplacian coordinate caused by offsets"""
    reg_offset_lap_relax_coef: float = 0.1  #0.03  # NOTE: tuned
    """The coefficient for relaxing reg_offset_lap for the regions specified"""
    reg_offset_lap_relax_for: tuple[str, ...] = ("hair", "ears")
    """Relax the offset loss for the regions specified"""

    reg_offset_rigid: Optional[float] = 3e3  # NOTE: tuned
    """Regularize the the offsets to be as-rigid-as-possible"""
    reg_offset_rigid_for: tuple[str, ...] = ("left_ear", "right_ear", "neck", "left_eye", "right_eye", "lips_tight")
    """Regularize the the offsets to be as-rigid-as-possible for the regions specified"""

    reg_offset_dynamic: Optional[float] = 3e5  # NOTE: tuned
    """Regularize the dynamic offsets to be temporally smooth"""

    blur_iter: int = 0
    """The number of iterations for blurring vertex weights"""
    
    reg_offset_eyes: Optional[float] = None  #1e3  # NOTE: tuned
    """Regularize offsets of the eyes region to avoid penetration """

    smooth_trans: float = 3e2  # NOTE: tuned
    """global translation"""
    smooth_rot: float = 3e1  # NOTE: tuned
    """global rotation"""

    smooth_neck: float = 3e1  # NOTE: tuned
    """neck joint"""
    smooth_neck_base: float = 3e1  # NOTE: tuned
    """neck base joint"""
    smooth_jaw: float = 1e-1
    """jaw joint"""
    smooth_eyes: float = 0
    """eyes joints"""

    prior_neck: float = 3e-1  # NOTE: tuned
    """Regularize the neck joint towards neutral"""
    prior_neck_base: float = 3e-1
    """Regularize the neck base joint towards neutral"""
    prior_jaw: float = 3e-1  # NOTE: tuned
    """Regularize the jaw joint towards neutral"""
    prior_eyes: float = 3e-2  # NOTE: tuned
    """Regularize the eyes joints towards neutral"""
    

@dataclass()
class LogConfig(Config):
    interval_scalar: Optional[int] = 100
    """The step interval of scalar logging. Using an interval of stage_tracking.num_steps // 5 unless specified."""
    interval_media: Optional[int] = 500
    """The step interval of media logging. Using an interval of stage_tracking.num_steps unless specified."""
    image_format: Literal['jpg', 'png'] = 'jpg'
    """Output image format"""
    view_indices: Tuple[int, ...] = ()
    """Manually specify the view indices for log"""
    max_num_views: int = 3
    """The maximum number of views for log"""
    stack_views_in_rows: bool = True


@dataclass()
class ExperimentConfig(Config):
    output_folder: Path = Path('output')
    name: str = 'track'
    reuse_landmarks: bool = True
    keyframes: Tuple[int, ...] = tuple()
    sub_steps: int = 1


@dataclass()
class StageLmkRigidConfig(Config):
    num_epochs: int = 1
    num_steps: int = 500

@dataclass()
class StageLmkConfig(Config):
    num_epochs: int = 1
    num_steps: int = 500

@dataclass()
class StageRgbTextureConfig(Config):
    num_epochs: int = 1
    num_steps: int = 500
    align_texture_except: tuple[str, ...] = ("hair", "boundary", "neck")
    """Align the texture space of FLAME to the image, except for the regions specified (NOTE: different from other stages)"""
    align_boundary_except: tuple[str, ...] = ("hair", "boundary")
    """Align the boundary of FLAME to the image, except for the regions specified"""
    reg_tex_var: Optional[float] = None
    """Overwrite the default value in LossWeightConfig"""

@dataclass()
class StageRgbConfig(Config):
    num_epochs: int = 1
    num_steps: int = 500
    align_texture_except: tuple[str, ...] = ("hair", "boundary", "neck")
    """Align the texture space of FLAME to the image, except for the regions specified (NOTE: different from other stages)"""
    align_boundary_except: tuple[str, ...] = ("hair", "bottomline")  # detach bottomline instead of boundary to better align shoulders
    """Align the boundary of FLAME to the image, except for the regions specified"""

@dataclass()
class StageRgbOffsetConfig(Config):
    num_epochs: int = 1
    num_steps: int = 500
    align_texture_except: tuple[str, ...] = ("hair", "boundary", "neck")
    """Align the texture space of FLAME to the image, except for the regions specified (NOTE: different from other stages)"""
    align_boundary_except: tuple[str, ...] = ("bottomline",)  # detach bottomline instead of boundary to better align shoulders
    """Align the boundary of FLAME to the image, except for the regions specified (NOTE: different from other stages)"""

@dataclass()
class StageSequentialTrackingConfig(Config):
    num_epochs: int = 1
    num_steps: int = 50
    align_texture_except: tuple[str, ...] = ("boundary",)
    """Align the texture space of FLAME to the image, except for the regions specified"""
    align_boundary_except: tuple[str, ...] = ("boundary",)
    """Align the boundary of FLAME to the image, except for the regions specified (NOTE: different from other stages)"""

@dataclass()
class StageGlobalTrackingConfig(Config):
    num_epochs: int = 30
    num_steps: int = 50
    align_texture_except: tuple[str, ...] = ("boundary",)
    """Align the texture space of FLAME to the image, except for the regions specified"""
    align_boundary_except: tuple[str, ...] = ("boundary",)
    """Align the boundary of FLAME to the image, except for the regions specified (NOTE: different from other stages)"""
    disable_landmark_loss: bool = False
    # """Not use the landmark loss at the end of the stage"""

@dataclass()
class StageConfig(Config):
    lmk_rigid: StageLmkRigidConfig
    lmk: StageLmkConfig
    rgb_texture: StageRgbTextureConfig
    rgb: StageRgbConfig  # to be updated in __post_init__
    rgb_offset: StageRgbOffsetConfig  # to be updated in __post_init__
    rgb_sequential_tracking: StageSequentialTrackingConfig  # to be updated in __post_init__
    rgb_global_tracking: StageGlobalTrackingConfig  # to be updated in __post_init__

    
@dataclass()
class BaseTrackingConfig(Config):
    data: DataConfig
    model: ModelConfig
    render: RenderConfig
    log: LogConfig
    exp: ExperimentConfig
    lr: LearningRateConfig
    w: LossWeightConfig
    stages: StageConfig

    begin_stage: Optional[str] = None
    """Begin from the specified stage for debugging"""
    begin_frame_idx: int = 0
    """Begin from the specified frame index for debugging"""
    async_func: bool = True
    """Allow asynchronous function calls for speed up"""
    device: Literal['cuda', 'cpu'] = 'cuda'

    def __post_init__(self):
        for stage_name in ['rgb_texture', 'rgb', 'rgb_offset', 'rgb_sequential_tracking', 'rgb_global_tracking']:
            stage = self.stages[stage_name]
            stage.align_texture_except = tuple(list(stage.align_texture_except) + list(self.model.occluded))
            stage.align_boundary_except = tuple(list(stage.align_boundary_except) + list(self.model.occluded))

        if self.begin_stage is not None:
            skip = True
            for stage_name in ['lmk_rigid', 'lmk', 'rgb_texture', 'rgb', 'rgb_offset', 'rgb_sequential_tracking', 'rgb_global_tracking']:  
                """rgb_tracking is not included because it is the last stage"""
                if stage_name == self.begin_stage:
                    skip = False
                
                if skip:
                    self.stages[stage_name].num_steps = 0


if __name__ == "__main__":
    config = tyro.cli(BaseTrackingConfig)
    print(tyro.to_yaml(config))