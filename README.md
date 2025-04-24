# GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians

<div align="center"> 
  <img src="media/demo.gif">

  <br>

  [project](https://shenhanqian.github.io/gaussian-avatars) / [arxiv](http://arxiv.org/abs/2312.02069) / [video](https://www.youtube.com/watch?v=lVEY78RwU_I) / [face tracker](https://github.com/ShenhanQian/VHAP) / [bibtex](https://shenhanqian.github.io/raw.html?filePath=/assets/2023-12-04-gaussian-avatars/bibtex.bib)
</div>

## Licenses

This work is made available under [CC-BY-NC-SA-4.0](./LICENSE.md) and is subject to the following statement:

> Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to this software and related documentation. Any commercial use, reproduction, disclosure or distribution of this software and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.

This project uses [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), which carries its [original license](./LICENSE_GS.md).
The GUI is inspired by [INSTA](https://github.com/Zielon/INSTA). 
The mesh rendering operations are adapted from [NVDiffRec](https://github.com/NVlabs/nvdiffrec) and [NVDiffRast](https://github.com/NVlabs/nvdiffrast). 

![Method](media/method.jpg)

## Setup

### [1. Installation](doc/installation.md)

### [2. Download](doc/download.md)

## Usage

### 0. Demo
You can play with a trained GaussianAvatar without downloading the dataset:
```shell
python local_viewer.py --point_path media/306/point_cloud.ply
```

### 1. Training

```shell
SUBJECT=306

python train.py \
-s data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--eval --bind_to_mesh --white_background --port 60000
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

- `--source_path` / `-s`

    Path to the source directory containing a COLMAP or Synthetic NeRF data set.

- `--model_path` / `-m`

    Path where the trained model should be stored (```output/<random>``` by default).

- `--eval`

   Add this flag to use a training/val/test split for evaluation. Otherwise, all images are used for training.

- `--bind_to_mesh`

  Add this flag to bind 3D Gaussians to a driving mesh, e.g., FLAME.

- `--resolution` / `-r`

  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**

- `--white_background` / `-w`

  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.

- `--sh_degree`

    Order of spherical harmonics to be used (no larger than 3). ```3``` by default.

- `--iterations`

  Number of total iterations to train for, ```30_000``` by default.

- `--port`

  Port to use for GUI server, ```60000``` by default.

</details>

> [!NOTE]
> During training, a complete evaluation are conducted on both the validation set (novel-view synthesis) and test set (self-reenactment) every `--interval` iterations. You can check the metrics in the commandline or Tensorboard. The metrics are computed on all images, although we only save partial images in Tensorboard.

### 2. Interactive Viewers

#### Remote Viewer

![remote viewer](media/remote_viewer.png)

During training, one can monitor the training progress with the remote viewer

```shell
python remote_viewer.py --port 60000
```

> [!NOTE]
> - The remote viewer can slow down training a lot. You may want to close it or check "pause rendering" when not viewing.
>
> - The viewer could get frozen and disconnected the first time you enable "show mesh". You can try switching it on and off or simply wait for a few seconds.

#### Local Viewer

![local viewer](media/local_viewer.png)

After training, one can load and render the optimized 3D Gaussians with the local viewer

```shell
SUBJECT=306
ITER=300000

python local_viewer.py \
--point_path output/UNION10EMOEXP_${SUBJECT}_eval_600k/point_cloud/iteration_${ITER}/point_cloud.ply
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

- `--point_path`

  Path to the gaussian splatting file (ply)

- `--motion_path`

  Path to the motion file (npz). You only need this if you want to load a different motion sequence than the original one for training.

</details>

> [!WARNING]
> The viewer is implemented in Python, making development convenient but not ideal for performance benchmarking. As such, please avoid using the viewer to measure the rendering frame rate of our method. Instead, use the [FPS benchmark script](https://github.com/ShenhanQian/GaussianAvatars/blob/main/doc/offline_render.md#fps-benchmark) for accurate performance evaluation.


### [3. Offline Rendering](doc/offline_render.md)

## Cite

If you find our paper or code useful in your research, please cite with the following BibTeX entry:
```bibtex
@inproceedings{qian2024gaussianavatars,
  title={Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians},
  author={Qian, Shenhan and Kirschstein, Tobias and Schoneveld, Liam and Davoli, Davide and Giebenhain, Simon and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20299--20309},
  year={2024}
}
```
