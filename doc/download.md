## Download

### FLAME Model

Our code and the pre-processed data relies on FLAME 2023. Please download [original assets](https://flame.is.tue.mpg.de/download.php) to the following paths:

- FLAME 2023 (versions w/ jaw rotation) -> `flame_model/assets/flame/flame2023.pkl`
- FLAME Vertex Masks -> `flame_model/assets/flame/FLAME_masks.pkl`

> **NOTE:** If you need to run our method with FLAME 2020, please download the corresponding model to`flame_model/assets/flame/generic_model.pkl`, and update `FLAME_MODEL_PATH` in `flame_model/flame.py` accordingly. Meanwhile, the FLAME tracking results should also be based on FLAME 2020 in this case.

<br>

### Human Head Video Data

#### 1. Preprocessed NeRSemble Dataset

In our paper, we use 9 subjects from the NeRSemble dataset. You can download the pre-processed data from

- [LRZ](https://syncandshare.lrz.de/getlink/fiRXRYvdGQoC162RZDDaZc/release) (directly accessible)
- [OneDrive](https://tumde-my.sharepoint.com/:f:/g/personal/shenhan_qian_tum_de/EtgO7DSNVzNKuYMRQeL4PE0BqMsTwdpQ09puewDLQBz87A) (request [here](https://forms.gle/dPEJx5DNvmhTm2Ry5)).

#### 2. Custom data

You can use our latest head-tracking pipeline, [VHAP](https://github.com/ShenhanQian/VHAP), to preprocess your custom data.
