from mvht.util.log import get_logger
from mvht.data.video import frame2id

from typing import Literal
from tqdm import tqdm

import face_alignment
import numpy as np
import matplotlib.path as mpltPath

from fdlite import (
    FaceDetection,
    FaceLandmark,
    face_detection_to_roi,
    IrisLandmark,
    iris_roi_from_face_landmarks,
)

logger = get_logger(__name__)


class LandmarkDetector:

    IMAGE_FILE_NAME = "image_0000.png"
    LMK_FILE_NAME = "keypoints_static_0000.json"

    def __init__(
        self,
        face_detector=Literal["sfd", "blazeface"],
    ):
        """
        Creates dataset_path where all results are stored
        :param video_path: path to video file
        :param dataset_path: path to results directory
        """

        logger.info("Initialize FaceAlignment module...")
        # 68 facial landmark detector
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_HALF_D, 
            face_detector=face_detector,
            flip_input=True, 
            device="cuda"
        )

    def detect_single_image(self, img):
        bbox = self.fa.face_detector.detect_from_image(img)

        if len(bbox) == 0:
            lmks = np.zeros([68, 3]) - 1  # set to -1 when landmarks is inavailable

        else:
            if len(bbox) > 1:
                # if multiple boxes detected, use the one with highest confidence
                bbox = [bbox[np.argmax(np.array(bbox)[:, -1])]]

            lmks = self.fa.get_landmarks_from_image(img, detected_faces=bbox)[0]
            lmks = np.concatenate([lmks, np.ones_like(lmks[:, :1])], axis=1)

            if (lmks[:, :2] == -1).sum() > 0:
                lmks[:, 2:] = 0.0
            else:
                lmks[:, 2:] = 1.0

            h, w = img.shape[:2]
            lmks[:, 0] /= w
            lmks[:, 1] /= h
            bbox[0][[0, 2]] /= w
            bbox[0][[1, 3]] /= h
        return bbox, lmks

    def detect_dataset(self, dataloader):
        """
        Annotates each frame with 68 facial landmarks
        :return: dict mapping frame number to landmarks numpy array and the same thing for bboxes
        """
        landmarks = {}
        bboxes = {}

        logger.info("Begin annotating landmarks...")
        for item in tqdm(dataloader):
            timestep_id = item["timestep_id"][0]
            camera_id = item["camera_id"][0]
            scale_factor = item["scale_factor"][0]

            logger.info(
                f"Annotate facial landmarks for timestep: {timestep_id}, camera: {camera_id}"
            )
            img = item["rgb"][0].numpy()
            
            bbox, lmks = self.detect_single_image(img)

            if len(bbox) == 0:
                logger.error(
                    f"No bbox found for frame: {timestep_id}, camera: {camera_id}. Setting landmarks to all -1."
                )

            if camera_id not in landmarks:
                landmarks[camera_id] = {}
            if camera_id not in bboxes:
                bboxes[camera_id] = {}
            landmarks[camera_id][timestep_id] = lmks
            bboxes[camera_id][timestep_id] = bbox[0] if len(bbox) > 0 else np.zeros(5) - 1
        return landmarks, bboxes

    def annotate_iris_landmarks(self, dataloader):
        """
        Annotates each frame with 2 iris landmarks
        :return: dict mapping frame number to landmarks numpy array
        """

        # iris detector
        detect_faces = FaceDetection()
        detect_face_landmarks = FaceLandmark()
        detect_iris_landmarks = IrisLandmark()

        landmarks = {}

        for item in tqdm(dataloader):
            timestep_id = item["timestep_id"][0]
            camera_id = item["camera_id"][0]
            scale_factor = item["scale_factor"][0]
            if timestep_id not in landmarks:
                landmarks[timestep_id] = {}
            logger.info(
                f"Annotate iris landmarks for timestep: {timestep_id}, camera: {camera_id}"
            )

            img = item["rgb"][0].numpy()

            height, width = img.shape[:2]
            img_size = (width, height)

            face_detections = detect_faces(img)
            if len(face_detections) != 1:
                logger.error("Empty iris landmarks (type 1)")
                landmarks[timestep_id][camera_id] = None
            else:
                for face_detection in face_detections:
                    try:
                        face_roi = face_detection_to_roi(face_detection, img_size)
                    except ValueError:
                        logger.error("Empty iris landmarks (type 2)")
                        landmarks[timestep_id][camera_id] = None
                        break

                    face_landmarks = detect_face_landmarks(img, face_roi)
                    if len(face_landmarks) == 0:
                        logger.error("Empty iris landmarks (type 3)")
                        landmarks[timestep_id][camera_id] = None
                        break

                    iris_rois = iris_roi_from_face_landmarks(face_landmarks, img_size)

                    if len(iris_rois) != 2:
                        logger.error("Empty iris landmarks (type 4)")
                        landmarks[timestep_id][camera_id] = None
                        break

                    lmks = []
                    for iris_roi in iris_rois[::-1]:
                        try:
                            iris_landmarks = detect_iris_landmarks(img, iris_roi).iris[
                                0:1
                            ]
                        except np.linalg.LinAlgError:
                            logger.error("Failed to get iris landmarks")
                            landmarks[timestep_id][camera_id] = None
                            break

                        for landmark in iris_landmarks:
                            lmks.append([landmark.x * width, landmark.y * height, 1.0])

                    lmks = np.array(lmks, dtype=np.float32)

                    h, w = img.shape[:2]
                    lmks[:, 0] /= w
                    lmks[:, 1] /= h

                    landmarks[timestep_id][camera_id] = lmks

        return landmarks

    def iris_consistency(self, lm_iris, lm_eye):
        """
        Checks if landmarks for eye and iris are consistent
        :param lm_iris:
        :param lm_eye:
        :return:
        """
        lm_iris = lm_iris[:, :2]
        lm_eye = lm_eye[:, :2]

        polygon_eye = mpltPath.Path(lm_eye)
        valid = polygon_eye.contains_points(lm_iris)

        return valid[0]

    def annotate_landmarks(self, dataloader, add_iris=False):
        """
        Annotates each frame with landmarks for face and iris. Assumes frames have been extracted
        :param add_iris:
        :return:
        """
        lmks_face, bboxes_faces = self.detect_dataset(dataloader)

        if add_iris:
            lmks_iris = self.annotate_iris_landmarks(dataloader)

            # check conistency of iris landmarks and facial keypoints
            for camera_id, lmk_face_camera in lmks_face.items():
                for timestep_id in lmk_face_camera.keys():

                    discard_iris_lmks = False
                    bboxes_face_i = bboxes_faces[camera_id][timestep_id]
                    if bboxes_face_i is not None:
                        lmks_face_i = lmks_face[camera_id][timestep_id]
                        lmks_iris_i = lmks_iris[camera_id][timestep_id]
                        if lmks_iris_i is not None:

                            # validate iris landmarks
                            left_face = lmks_face_i[36:42]
                            right_face = lmks_face_i[42:48]

                            right_iris = lmks_iris_i[:1]
                            left_iris = lmks_iris_i[1:]

                            if not (
                                self.iris_consistency(left_iris, left_face)
                                and self.iris_consistency(right_iris, right_face)
                            ):
                                logger.error(
                                    f"Inconsistent iris landmarks for timestep: {timestep_id}, camera: {camera_id}"
                                )
                                discard_iris_lmks = True
                        else:
                            logger.error(
                                f"No iris landmarks detected for timestep: {timestep_id}, camera: {camera_id}"
                            )
                            discard_iris_lmks = True

                    else:
                        logger.error(
                            f"Discarding iris landmarks because no face landmark is available for timestep: {timestep_id}, camera: {camera_id}"
                        )
                        discard_iris_lmks = True

                    if discard_iris_lmks:
                        lmks_iris[timestep_id][camera_id] = (
                            np.zeros([2, 3]) - 1
                        )  # set to -1 for inconsistent iris landmarks

        # construct final json
        for camera_id, lmk_face_camera in lmks_face.items():
            bounding_box = []
            face_landmark_2d = []
            iris_landmark_2d = []
            for timestep_id in lmk_face_camera.keys():
                bounding_box.append(bboxes_faces[camera_id][timestep_id][None])
                face_landmark_2d.append(lmks_face[camera_id][timestep_id][None])

                if add_iris:
                    iris_landmark_2d.append(lmks_iris[camera_id][timestep_id][None])

            lmk_dict = {
                "bounding_box": bounding_box,
                "face_landmark_2d": face_landmark_2d,
            }
            if len(iris_landmark_2d) > 0:
                lmk_dict["iris_landmark_2d"] = iris_landmark_2d

            for k, v in lmk_dict.items():
                if len(v) > 0:
                    lmk_dict[k] = np.concatenate(v, axis=0)
            out_path = dataloader.dataset.get_property_path(
                "face-alignment", camera_id=camera_id
            )
            logger.info(f"Saving landmarks to: {out_path}")
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True)
            np.savez(out_path, **lmk_dict)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from torch.utils.data import DataLoader
    from mvht.data.multi_view_head_dataset import MultiViewHeadDataset

    logger = get_logger(__name__, root=True)

    parser = ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    args = parser.parse_args()

    dataset = MultiViewHeadDataset(
        root_folder=args.root_folder,
        subject=args.subject,
        sequence=args.sequence,
        scale_factor=args.scale_factor,
    )
    dataset.items = dataset.items[:2]

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    detector = LandmarkDetector()
    detector.annotate_landmarks(dataloader)
