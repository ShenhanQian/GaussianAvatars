# Multi-view Head Tracker

This is a reference repo for head tracking used by GaussianAvatars. **It is only for reference of technical details rather than running.**

## License

Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to this software and related documentation. Any commercial use, reproduction, disclosure or distribution of this software and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.

This repo is developed on top of [VHT](https://github.com/philgras/video-head-tracker), which carries its original license. The mesh rendering operations are adapted from NVDiffRec and NVDiffRast.

This work is made available under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.

## Installation

```shell
pip install -e .
```

For face landmark detection, [face-alignment](https://github.com/1adrianb/face-alignment) is the easiest to install while [STAR](https://github.com/ShenhanQian/STAR/) is more precise.

## Run

```shell
SUBJECT="306"
SEQUENCE="EMO-1"
NERSEMBLE_DIR="path-to-nersemble-dateset"

python mvht/nersemble_tracking.py --data.root_folder $NERSEMBLE_DIR \
--exp.name output/${SUBJECT}_${SEQUENCE}_v16_DS2-0.5x-wBg_lmkSTAR_teethV3_SMOOTHtrans3e2-rot+neck3e1_PRIOReyes3e-2_albedoTV1e5-res1e1_SH-regDiffuse1e2_offsetS-3e1-lap1e6-relax0.1-rigid3e3_ep30 \
--data.subject $SUBJECT --data.sequence $SEQUENCE \
--data.n_downsample_rgb 2 --data.scale_factor 0.5
```

## Troubleshooting

### FFMPEG

```
MovieWriter stderr:
[libopenh264 @ 0x55699c4ce400] Incorrect library version loaded
Error initializing output stream 0:0 -- Error while opening encoder for output stream #0:0 - maybe incorrect parameters such as bit_rate, rate, width or height
```

Solution:

``` shell
# upgrade ffmpeg
conda install ffmpeg
```

### EGL

  ```
  fatal error: EGL/egl.h: No such file or directory
  ```
  Solution:
  ```
  sudo apt-get install libgles2-mesa-dev
  ```
