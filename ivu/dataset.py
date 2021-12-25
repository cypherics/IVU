import os
from collections import defaultdict

import cv2
import numpy as np

from py_oneliner import one_liner
from scipy.spatial.distance import squareform, pdist

from ivu.pose_estimator.base_pose_estimator import PoseEstimationFailedError
from ivu.pose_estimator.media_pipe_estimator import get_media_pipe_pose_estimator
from ivu.pose_estimator.pose_landmarks import (
    Pose16LandmarksBodyModel,
    landmarks_to_embedding,
)
from ivu.utils import (
    folder_generator,
    save_pickle,
    read_video,
)


class Data:
    def __init__(self):
        self._meta = defaultdict(list)

    def add_data(self, **kwargs):
        for key, value in kwargs.items():
            self._meta[key].append(value)

    def get_meta(self):
        return self._meta

    def store(self, save_dir: str = None):
        save_dir = os.getcwd() if save_dir is None else save_dir
        pth = os.path.join(save_dir, "key_points.pickle")
        save_pickle(self._meta, pth)


def get_pose_data_from_rgb_frame(frame, pose_estimator):
    body_key_points = pose_estimator.get_key_points_from_image(frame)
    distance_matrix = squareform(pdist(np.array(body_key_points)))

    normalized_body_key_points = landmarks_to_embedding(
        body_key_points, Pose16LandmarksBodyModel
    )
    normalized_distance_matrix = squareform(
        pdist(np.array(normalized_body_key_points[0]))
    )

    return (
        body_key_points,
        distance_matrix,
        normalized_body_key_points,
        normalized_distance_matrix,
    )


def generate_data_over_images(
    data_set_dir: str,
    pose_estimator_complexity=1,
    use_pose_estimator_over_static_image=True,
) -> Data:
    pose_estimator = get_media_pipe_pose_estimator(
        complexity=pose_estimator_complexity,
        static_image_mode=use_pose_estimator_over_static_image,
    )

    data = Data()
    for class_label_index, class_label in folder_generator(data_set_dir):
        files = os.listdir(os.path.join(data_set_dir, class_label))
        for iterator, file in enumerate(files):
            file_path = os.path.join(*[data_set_dir, class_label, file])
            one_liner.one_line(
                tag=f"[PROGRESS: {iterator + 1}/{len(files)}]",
                tag_data=f"[CURRENT FILE : {file}] [LABEL: {class_label}]",
                to_reset_data=True,
                tag_color="red",
                tag_data_color="yellow",
            )

            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            try:

                (
                    body_key_points,
                    distance_matrix,
                    normalized_body_key_points,
                    normalized_distance_matrix,
                ) = get_pose_data_from_rgb_frame(image, pose_estimator)
                data.add_data(
                    key_points=body_key_points,
                    normalized_key_points=normalized_body_key_points[0],
                    normalized_distance_matrix=normalized_distance_matrix[
                        np.triu_indices(normalized_distance_matrix.shape[0], k=1)
                    ],
                    distance_matrix=distance_matrix[
                        np.triu_indices(distance_matrix.shape[0], k=1)
                    ],
                    class_label=class_label,
                    class_label_index=class_label_index,
                    file=file,
                )

            except PoseEstimationFailedError:
                pass

    return data


def generate_data_over_videos(
    data_set_dir: str,
    pose_estimator_complexity=1,
    use_pose_estimator_over_static_image=True,
    frame_width=-1,
    frame_height=-1,
) -> Data:
    pose_estimator = get_media_pipe_pose_estimator(
        complexity=pose_estimator_complexity,
        static_image_mode=use_pose_estimator_over_static_image,
    )

    data = Data()
    for class_label_index, class_label in folder_generator(data_set_dir):
        files = os.listdir(os.path.join(data_set_dir, class_label))
        for iterator, file in enumerate(files):
            file_path = os.path.join(*[data_set_dir, class_label, file])
            vr = read_video(file_path, width=frame_width, height=frame_height)
            for frame in range(len(vr)):
                one_liner.one_line(
                    tag=f"PROGRESS [VIDEOS: {iterator + 1}/{len(files)}] [CURRENT FILE : {file}] [LABEL: {class_label}]",
                    tag_data=f"[FRAMES : {frame + 1}/{len(vr)}]",
                    to_reset_data=True,
                    tag_color="red",
                    tag_data_color="red",
                )

                try:
                    (
                        body_key_points,
                        distance_matrix,
                        normalized_body_key_points,
                        normalized_distance_matrix,
                    ) = get_pose_data_from_rgb_frame(
                        vr[frame].asnumpy(), pose_estimator
                    )
                    data.add_data(
                        key_points=body_key_points,
                        normalized_key_points=normalized_body_key_points[0],
                        normalized_distance_matrix=normalized_distance_matrix[
                            np.triu_indices(normalized_distance_matrix.shape[0], k=1)
                        ],
                        distance_matrix=distance_matrix[
                            np.triu_indices(distance_matrix.shape[0], k=1)
                        ],
                        class_label=class_label,
                        class_label_index=class_label_index,
                        file=file,
                        frame_number=frame,
                        frame_details=f"{class_label_index}_{iterator}_{frame}",
                    )
                except PoseEstimationFailedError:
                    pass

    return data


def create_sequence_data_set_over_videos(data_set_path, frame_count):
    pass


def data_set_over_images(
    data_set_dir: str,
    save_dir: str,
    pose_estimator_complexity=1,
    use_pose_estimator_over_static_image=True,
):
    data = generate_data_over_images(
        data_set_dir=data_set_dir,
        pose_estimator_complexity=pose_estimator_complexity,
        use_pose_estimator_over_static_image=use_pose_estimator_over_static_image,
    )
    data.store(save_dir=save_dir)


def data_set_over_videos(
    data_set_dir: str,
    save_dir: str,
    pose_estimator_complexity=1,
    use_pose_estimator_over_static_image=True,
    width=-1,
    height=-1,
):
    data = generate_data_over_videos(
        data_set_dir=data_set_dir,
        pose_estimator_complexity=pose_estimator_complexity,
        use_pose_estimator_over_static_image=use_pose_estimator_over_static_image,
        frame_width=width,
        frame_height=height,
    )
    data.store(save_dir=save_dir)
