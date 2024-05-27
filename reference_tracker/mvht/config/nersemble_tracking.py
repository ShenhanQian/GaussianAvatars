# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from dataclasses import dataclass
from typing import Optional, Literal
import tyro

from mvht.config.base import DataConfig, BaseTrackingConfig
from mvht.util.log import get_logger


logger = get_logger(__name__)


@dataclass()
class NersembleDataConfig(DataConfig):
    subject: str = ""
    sequence: str = ""
    calibrated: bool = True
    use_color_correction: bool = True
    use_alpha_map: bool = False
    use_landmark: bool = True
    landmark_source: Optional[Literal["face-alignment", "pipnet", 'star']] = "star"

    
@dataclass()
class NersembleTrackingConfig(BaseTrackingConfig):
    data: NersembleDataConfig

    def __post_init__(self):
        occluded_table = {
            '018': ('neck_lower',),
            '218': ('neck_lower',),
            '251': ('neck_lower', 'boundary'),
            '253': ('neck_lower',),
        }
        if self.data.subject in occluded_table:
            logger.info(f"Automatically setting cfg.model.occluded to {occluded_table[self.data.subject]}")
            self.model.occluded = occluded_table[self.data.subject]
        super().__post_init__()


if __name__ == "__main__":
    config = tyro.cli(NersembleTrackingConfig)
    print(tyro.to_yaml(config))