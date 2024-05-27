# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


import tyro

from mvht.config.nersemble_tracking import NersembleTrackingConfig
from mvht.util.log import get_logger
from mvht.model.tracking import GlobalTracker

logger = get_logger("mvht", root=True)


def main():
    tyro.extras.set_accent_color("bright_yellow")
    cfg = tyro.cli(NersembleTrackingConfig)

    tracker = GlobalTracker(cfg)
    tracker.optimize()


if __name__ == "__main__":
    main()

