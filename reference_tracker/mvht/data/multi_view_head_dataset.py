# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import os
from copy import deepcopy
from typing import Optional, Literal
from pathlib import Path
from datetime import datetime
import json
import csv
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, default_collate
from pytorch3d.transforms import axis_angle_to_matrix
from mvht.util import camera
from mvht.util.log import get_logger


logger = get_logger(__name__)


class MultiViewHeadDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        subject: str,
        sequence: str,
        division: Optional[str] = None,
        subset: Optional[str] = None,
        n_downsample_rgb: Optional[int] = None,  # times of downsample for the rgb image
        scale_factor: float = 1.0,  # scale_factor of all resolutions and coordinates
        align_cameras_to_axes: bool = True,
        camera_coord_conversion: Optional[
            Literal["opencv->opengl", "opencv->pytorch3d"]
        ] = None,  # transformation between camera coordiante conventions
        extrinsic_type: Optional[Literal["w2c", "c2w"]] = "w2c",
        background_color: Optional[str] = None,
        use_color_correction = True,
        use_alpha_map: bool = False,
        use_landmark: bool = False,
        landmark_source: Optional[Literal["face-alignment", "pipnet", "star"]] = "star",
        use_flame: bool = False,
        img_to_tensor: bool = False,
        rgb_range_shift: bool = True,
        batchify_all_views: bool = False,
    ):
        """
        Args:
            root_folder: Path to dataset with the following directory layout
                <root_folder>/
                |
                |---<subject>/
                    |---calibration/
                    |   |---calibration_result.json
                    |
                    |---sequences/
                        |---<sequence>/
                            |---timesteps/
                            |   |---frames_00000
                            |       |---images/
                            |           |---<camera_id>.png
                            |
                            |---annotations/
                                |---color_correction/
                                |    |---<camera_id>.npy
                                |
                                |---landmarks2D/
                                     |---face-alignment/
                                     |    |---<camera_id>.npz
                                     |
                                     |---STAR/
                                          |---<camera_id>.npz


        """

        super().__init__()
        self.root_folder = Path(root_folder)
        self.division = division
        self.subject = subject
        self.sequence = sequence
        self.n_downsample_rgb = n_downsample_rgb
        self.scale_factor = scale_factor
        self.align_cameras_to_axes = align_cameras_to_axes
        self.camera_coord_conversion = camera_coord_conversion
        self.extrinsic_type = extrinsic_type
        self.background_color = background_color
        self.use_color_correction = use_color_correction
        self.use_alpha_map = use_alpha_map
        self.use_landmark = use_landmark
        self.landmark_source = landmark_source
        self.use_flame = use_flame
        self.img_to_tensor = img_to_tensor
        self.rgb_range_shift = rgb_range_shift
        self.batchify_all_views = batchify_all_views

        logger.info(f"Initializing dataset...")
        sequences = [
            "EMO-1-shout+laugh", "EMO-2-surprise+fear", "EMO-3-angry+sad", "EMO-4-disgust+happy", 
            "EXP-1-head", "EXP-2-eyes", "EXP-3-cheeks+nose", "EXP-4-lips", "EXP-5-mouth", 
            "EXP-6-tongue-1", "EXP-7-tongue-2", "EXP-8-jaw-1", "EXP-9-jaw-2", 
            "FREE", 
            "SEN-01-cramp_small_danger", "SEN-02-same_phrase_thirty_times", 
            "SEN-03-pluck_bright_rose", "SEN-04-two_plus_seven", 
            "SEN-05-glow_eyes_sweet_girl", "SEN-06-problems_wise_chief", 
            "SEN-07-fond_note_fried", "SEN-08-clothes_and_lodging", 
            "SEN-09-frown_events_bad", "SEN-10-port_strong_smokey"
        ]
        sequence = [s for s in sequences if sequence in s][0]
        logger.info(f"Subject: {subject}, sequence: {sequence}")

        self.properties = {
            "rgb": {
                "type": "image",
                "level": "view",
                "folder": f"images-{n_downsample_rgb}x"
                if n_downsample_rgb
                else "images",
                "suffix": "*",
            },
            "alpha_map": {
                "type": "image",
                "level": "view",
                "folder": "alpha_map",
                "suffix": "png",
            },
            "face-alignment": {
                "type": "array",
                "level": "view",
                "folder": "landmarks2D/face-alignment",
                "suffix": "npz",
            },
            "landmarks2D/STAR": {
                "type": "array",
                "level": "view",
                "folder": "landmarks2D/STAR",
                "suffix": "npz",
            },
            "landmarks2D/PIPnet": {
                "type": "array",
                "level": "view",
                "folder": "landmarks2D/PIPnet",
                "cam_id_style": "digit",
                "suffix": "npy",
            },
            "color_correction": {
                "type": "array",
                "level": "view",
                "folder": "color_correction",
                "cam_id_style": "digit",
                "suffix": "npy",
            },
            "flame_param": {
                "type": "array",
                "level": "timestep",
                "folder": "photometric_tracking",
                "suffix": "npz",
            },
        }

        # cameras
        self.camera_ids = [
            "cam_222200042",
            "cam_222200044",
            "cam_222200046",
            "cam_222200040",
            "cam_222200036",
            "cam_222200048",
            "cam_220700191",
            "cam_222200041",
            "cam_222200037",  # front view
            "cam_222200038",
            "cam_222200047",
            "cam_222200043",
            "cam_222200049",
            "cam_222200039",
            "cam_222200045",
            "cam_221501007",
        ]
        self.camera_params = self.load_camera_params(subject)
        cam_for_train = [8, 7, 9, 4, 10, 5, 13, 2, 12, 1, 14, 0]

        # timesteps
        self.sequence_path = self.root_folder / subject / "sequences" / sequence

        if "moria" in str(self.root_folder):
            self.timesteps_path = self.sequence_path
            self.properties['rgb']["folder"] += "-73fps"
        elif "doriath" in str(self.root_folder):
            self.timesteps_path = self.sequence_path / 'timesteps'
        else:
            raise NotImplementedError(f"Unknown root folder: {root_folder}")

        self.timestep_ids = [
            f for f in os.listdir(self.timesteps_path) if f.startswith("frame_")
        ]
        self.timestep_ids = np.sort(self.timestep_ids)
        self.timestep_indices = list(range(len(self.timestep_ids)))

        # data division
        if division is not None:
            if division == "train":
                self.camera_ids = [
                    self.camera_ids[i]
                    for i in range(len(self.camera_ids))
                    if i in cam_for_train
                ]
            elif division == "val":
                self.camera_ids = [
                    self.camera_ids[i]
                    for i in range(len(self.camera_ids))
                    if i not in cam_for_train
                ]
            elif division == "front-view":
                self.camera_ids = self.camera_ids[8:9]
            elif division == "side-view":
                self.camera_ids = self.camera_ids[0:1]
            elif division == "six-view":
                self.camera_ids = [self.camera_ids[i] for i in [0, 1, 7, 8, 14, 15]]
            else:
                raise NotImplementedError(f"Unknown division type: {division}")
            logger.info(f"division: {division}")

        if subset is not None:
            if 'ti' in subset:
                ti = self.get_number_after_prefix(subset, 'ti')
                self.timestep_indices = self.timestep_indices[ti:ti+1]
            elif 'tn' in subset:
                tn = self.get_number_after_prefix(subset, 'tn')
                tn_all = len(self.timestep_indices)
                tn = min(tn, tn_all)
                self.timestep_indices = self.timestep_indices[::tn_all // tn][:tn]
            elif 'ts' in subset:
                ts = self.get_number_after_prefix(subset, 'ts')
                self.timestep_indices = self.timestep_indices[::ts]
            if 'ci' in subset:
                ci = self.get_number_after_prefix(subset, 'ci')
                self.camera_ids = self.camera_ids[ci:ci+1]
            elif 'cn' in subset:
                cn = self.get_number_after_prefix(subset, 'cn')
                cn_all = len(self.camera_ids)
                cn = min(cn, cn_all)
                self.camera_ids = self.camera_ids[::cn_all // cn][:cn]
            elif 'cs' in subset:
                cs = self.get_number_after_prefix(subset, 'cs')
                self.camera_ids = self.camera_ids[::cs]
            
        logger.info(f"number of timesteps: {self.num_timesteps}, number of cameras: {self.num_cameras}")

        # collect
        self.items = []
        for fi, timestep_index in enumerate(self.timestep_indices):
            for ci, camera_id in enumerate(self.camera_ids):
                self.items.append(
                    {
                        "timestep_index": fi,  # new index after filtering
                        "timestep_index_original": timestep_index,  # original index
                        "timestep_id": self.timestep_ids[timestep_index],
                        "camera_index": ci,
                        "camera_id": camera_id,
                    }
                )
    
    @staticmethod
    def get_number_after_prefix(string, prefix):
        i = string.find(prefix)
        if i != -1:
            number_begin = i + len(prefix)
            assert number_begin < len(string), f"No number found behind prefix '{prefix}'"
            assert string[number_begin].isdigit(), f"No number found behind prefix '{prefix}'"

            non_digit_indices = [i for i, c in enumerate(string[number_begin:]) if not c.isdigit()]
            if len(non_digit_indices) > 0:
                number_end = number_begin + min(non_digit_indices)
                return int(string[number_begin:number_end])
            else:
                return int(string[number_begin:])
        else:
            return None

    def load_camera_params(self, subject):
        load_path = self.root_folder / subject / "calibration" / "calibration_result.json"

        if not load_path.exists():
            subject_calib = self.find_closest_calibrated_participant_id(subject)
            load_path = self.root_folder / subject_calib / "calibration" / "calibration_result.json"
        param = json.load(open(load_path))["params_result"]

        intr = param["intrinsics"][0]
        K = torch.eye(3)
        K[[0, 1, 0, 1], [0, 1, 2, 2]] = torch.tensor(
            [intr["fx"], intr["fy"], intr["cx"], intr["cy"]]
        )

        axis_angle = torch.tensor(param["rs"])
        R = axis_angle_to_matrix(axis_angle)
        T = torch.tensor(param["ts"])

        orientation = R.transpose(-1, -2)
        location = R.transpose(-1, -2) @ -T[..., None]

        # adjust how cameras distribute in the space with a global rotation
        if self.align_cameras_to_axes:
            orientation, location = camera.align_cameras_to_axes(
                orientation, location, target_convention="opengl"
            )

        # modify the local orientation of cameras to fit in different camera conventions
        if self.camera_coord_conversion is not None:
            orientation = camera.change_camera_coord_convention(
                self.camera_coord_conversion, orientation
            )

        c2w = torch.cat([orientation, location], dim=-1)  # camera-to-world transformation

        if self.extrinsic_type == "w2c":
            R = orientation.transpose(-1, -2)
            T = orientation.transpose(-1, -2) @ -location
            w2c = torch.cat([R, T], dim=-1)  # world-to-camera transformation
            extrinsic = w2c
        elif self.extrinsic_type == "c2w":
            extrinsic = c2w
        else:
            raise NotImplementedError(f"Unknown extrinsic type: {self.extrinsic_type}")

        camera_params = {}
        for i, camera_id in enumerate(self.camera_ids):
            camera_params[camera_id] = {"intrinsic": K, "extrinsic": extrinsic[i]}

        return camera_params
    
    def find_closest_calibrated_participant_id(self, subject):
        calibrated_participant_ids = [p.stem for p in self.root_folder.iterdir() if p.is_dir() and (p / 'calibration' / 'calibration_result.json').exists()]
        meta_data_path = self.root_folder / 'participants_meta_data.csv'

        with open(meta_data_path, encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=',')
            column_names = next(csv_reader)
            rows = []
            for row in csv_reader:
                rows.append(row)

        columns = list(zip(*rows))
        idx_timestamp = column_names.index("Timestamp")
        idx_participant_id = column_names.index("ID")

        participant_ids = columns[idx_participant_id]
        calibrated = [p_id in calibrated_participant_ids for p_id in participant_ids]

        timestamps = columns[idx_timestamp]
        timestamps = [datetime.strptime(timestamp, "%d/%m/%Y %H:%M:%S") for timestamp in timestamps]

        calibrated_timestamps = [timestamp for timestamp, is_calibrated in zip(timestamps, calibrated) if is_calibrated]
        calibrated_p_ids = [p_id for p_id, is_calibrated in zip(participant_ids, calibrated) if is_calibrated]

        i_row_reference_session = participant_ids.index(str(int(subject)))
        reference_timestamp = timestamps[i_row_reference_session]

        time_differences_to_calibrated = [abs(timestamp - reference_timestamp) for timestamp in calibrated_timestamps]
        idx_closest = np.argmin(time_differences_to_calibrated)
        closest_calibrated_participant_id = calibrated_p_ids[idx_closest]

        logger.info(
            f"Using calibration of {closest_calibrated_participant_id} instead of requested {subject} ({min(time_differences_to_calibrated)} apart)")
        return closest_calibrated_participant_id
    
    def __len__(self):
        if self.batchify_all_views:
            return self.num_timesteps
        else:
            return len(self.items)

    def __getitem__(self, i):
        if self.batchify_all_views:
            return self.getitem_by_timestep(i)
        else:
            return self.getitem_single_image(i)

    def getitem_single_image(self, i):
        item = deepcopy(self.items[i])

        rgb_path = self.get_property_path("rgb", i)
        item["rgb"] = np.array(Image.open(rgb_path))

        camera_id = self.items[i]["camera_id"]
        camera_param = self.camera_params[camera_id]
        item["intrinsic"] = camera_param["intrinsic"].clone()
        item["extrinsic"] = camera_param["extrinsic"].clone()

        if self.use_color_correction:
            color_correction_path = self.get_property_path("color_correction", i)
            affine_color_transform = np.load(color_correction_path)
            rgb = item["rgb"] / 255
            rgb = rgb @ affine_color_transform[:3, :3] + affine_color_transform[np.newaxis, :3, 3]
            item["rgb"] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        if self.use_alpha_map:
            alpha_path = self.get_property_path("alpha_map", i)
            item["alpha_map"] = np.array(Image.open(alpha_path))

        if self.use_landmark:
            timestep_index = self.items[i]["timestep_index"]

            if self.landmark_source in ["face-alignment", "star"]:
                if self.landmark_source == "face-alignment":
                    landmark_path = self.get_property_path("face-alignment", i)
                elif self.landmark_source == "star": 
                    landmark_path = self.get_property_path("landmarks2D/STAR", i)
                landmark_npz = np.load(landmark_path)

                item["lmk2d"] = landmark_npz["face_landmark_2d"][timestep_index]  # (num_points, 3)
                if (item["lmk2d"][:, :2] == -1).sum() > 0:
                    item["lmk2d"][:, 2:] = 0.0
                else:
                    item["lmk2d"][:, 2:] = 1.0

                if "iris_landmark_2d" in landmark_npz:
                    item["lmk2d_iris"] = landmark_npz["iris_landmark_2d"][timestep_index]  # (num_points, 3)
                    if (item["lmk2d_iris"][:, :2] == -1).sum() > 0:
                        item["lmk2d_iris"][:, 2:] = 0.0  # drop both if anyone is inavailable
                    else:
                        item["lmk2d_iris"] = item["lmk2d_iris"][[1, 0]]  # swap left right iris
                        item["lmk2d_iris"][:, 2:] = 1.0

                item["bbox_2d"] = landmark_npz["bounding_box"][timestep_index]  # [x1, y1, x2, y2, score]
                if (item["bbox_2d"][:-1] == -1).sum() > 0:
                    item["bbox_2d"][-1:] = 0.0
                else:
                    item["bbox_2d"][-1:] = 1.0

            elif self.landmark_source == "pipnet":
                landmark_path = self.get_property_path("landmarks2D/PIPnet", i)
                landmark_npy = np.load(landmark_path)
                import ipdb; ipdb.set_trace()
                # item["lmk2d"] = None
            else:
                raise NotImplementedError(f"Unknown landmark source: {self.landmark_source}")

        if self.use_flame:
            item["flame_param"] = self.get_flame_param(i)

        item = self.apply_transforms(item)
        return item

    def getitem_by_timestep(self, timestep_index):
        begin = timestep_index * self.num_cameras
        indices = range(begin, begin + self.num_cameras)
        item = default_collate([self.getitem_single_image(i) for i in indices])

        item["num_cameras"] = self.num_cameras
        return item

    def apply_transforms(self, item):
        item = self.apply_scale_factor(item)
        item = self.apply_background_color(item)
        item = self.apply_to_tensor(item)
        return item

    def apply_to_tensor(self, item):
        if self.img_to_tensor:
            if "rgb" in item:
                item["rgb"] = F.to_tensor(item["rgb"])
                # if self.rgb_range_shift:
                #     item["rgb"] = (item["rgb"] - 0.5) / 0.5

            if "alpha_map" in item:
                item["alpha_map"] = F.to_tensor(item["alpha_map"])
        return item

    def apply_scale_factor(self, item):
        assert self.scale_factor <= 1.0

        if "rgb" in item:
            H, W, _ = item["rgb"].shape
            h, w = int(H * self.scale_factor), int(W * self.scale_factor)
            rgb = Image.fromarray(item["rgb"]).resize(
                (w, h), resample=Image.BILINEAR
            )
            item["rgb"] = np.array(rgb)
    
        # properties that are defined based on image size
        if "lmk2d" in item:
            item["lmk2d"][..., 0] *= w
            item["lmk2d"][..., 1] *= h
        
        if "lmk2d_iris" in item:
            item["lmk2d_iris"][..., 0] *= w
            item["lmk2d_iris"][..., 1] *= h

        if "bbox_2d" in item:
            item["bbox_2d"][[0, 2]] *= w
            item["bbox_2d"][[1, 3]] *= h

        # properties need to be scaled down when rgb is downsampled
        n_downsample_rgb = self.n_downsample_rgb if self.n_downsample_rgb else 1
        scale_factor = self.scale_factor / n_downsample_rgb
        item["scale_factor"] = scale_factor  # NOTE: not self.scale_factor
        if scale_factor < 1.0:
            if "intrinsic" in item:
                item["intrinsic"][:2] *= scale_factor
            if "alpha_map" in item:
                h, w = item["rgb"].shape[:2]
                alpha_map = Image.fromarray(item["alpha_map"]).resize(
                    (w, h), Image.Resampling.BILINEAR
                )
                item["alpha_map"] = np.array(alpha_map)
        return item

    def apply_background_color(self, item):
        if self.background_color is not None:
            assert (
                "alpha_map" in item
            ), "'alpha_map' is required to apply background color."
            fg = item["rgb"]
            if self.background_color == "white":
                bg = np.ones_like(fg) * 255
            elif self.background_color == "black":
                bg = np.zeros_like(fg)
            else:
                raise NotImplementedError(
                    f"Unknown background color: {self.background_color}."
                )

            w = item["alpha_map"][..., None] / 255
            img = (w * fg + (1 - w) * bg).astype(np.uint8)
            item["rgb"] = img
        return item

    def get_property_path(
        self,
        name,
        index: Optional[int] = None,
        timestep_id: Optional[str] = None,
        camera_id: Optional[str] = None,
    ):
        p = self.properties[name]
        type = p["type"]
        level = p["level"]
        folder = p["folder"] if "folder" in p else None
        suffix = p["suffix"]

        if type == 'array':
            path = self.sequence_path / "annotations"
        elif type == 'image':
            if timestep_id is None:
                assert index is not None, "index is required when timestep_id is not provided."
                timestep_id = self.items[index]["timestep_id"]
            path = self.timesteps_path / timestep_id
        else:
            raise NotImplementedError(f"Unknown property type: {type}")
        
        if folder is not None:
            path = path / folder

        if level == "sequence":
            path /= f"{name}.{suffix}"
        elif level == "timestep":
            path /= f"{name}.{suffix}"
        elif level == "view":
            if camera_id is None:
                assert (
                    index is not None), "index is required when camera_id is not provided."
                camera_id = self.items[index]["camera_id"]
            if "cam_id_style" in p and p["cam_id_style"] == "digit":
                camera_id = camera_id.split("_")[-1]

            path /= f"{camera_id}.{suffix}"
        else:
            raise NotImplementedError
        
        if "*" in str(path):
            try:
                path = next(path.parent.glob(path.name))
            except StopIteration:
                raise FileNotFoundError(f"File not found: {path}")
        return path
        
    def get_property_path_list(self, name):
        paths = []
        for i in range(len(self.items)):
            img_path = self.get_property_path(name, i)
            paths.append(img_path)
        return paths

    def get_flame_param(self, i):
        pass

    @property
    def num_timesteps(self):
        return len(self.timestep_ids)

    @property
    def num_cameras(self):
        return len(self.camera_ids)


if __name__ == "__main__":
    from tqdm import tqdm
    from dataclasses import dataclass
    import tyro
    from torch.utils.data import DataLoader

    @dataclass
    class Args:
        root_folder: str
        subject: str
        sequence: str
        use_landmark: bool = False
        batchify_all_views: bool = False

    args = tyro.cli(Args)

    dataset = MultiViewHeadDataset(
        root_folder=args.root_folder,
        subject=args.subject,
        sequence=args.sequence,
        use_landmark=args.use_landmark,
        batchify_all_views=args.batchify_all_views,
    )

    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())
    print(sample["rgb"].shape)

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=1)
    for item in tqdm(dataloader):
        pass
