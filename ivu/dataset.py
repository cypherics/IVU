import os
from collections import defaultdict

import cv2

from ivu.pose_estimator.base_pose_estimator import PoseEstimationFailedError
from ivu.pose_estimator.media_pipe_estimator import get_media_pipe_pose_estimator
from ivu.pose_estimator.pose_landmarks import (
    Pose16LandmarksBodyModel,
    landmarks_to_embedding,
)
from ivu.utils import default_dict_list_values_to_numpy_array, dataset_file_generator


def create_data_set_over_images(
    data_set_path: str,
    pose_estimator_complexity=1,
    use_pose_estimator_over_static_image=True,
):
    pose_estimator = get_media_pipe_pose_estimator(
        complexity=pose_estimator_complexity,
        static_image_mode=use_pose_estimator_over_static_image,
    )
    labels = os.listdir(data_set_path)

    poses = defaultdict(list)
    for file_path, class_label in dataset_file_generator(data_set_path, labels):
        image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

        try:

            body_key_points = pose_estimator.get_key_points_from_image(image)
            normalized_body_key_points = landmarks_to_embedding(
                body_key_points, Pose16LandmarksBodyModel
            )
            poses[class_label].append(normalized_body_key_points[0])

        except PoseEstimationFailedError:
            pass

    poses = default_dict_list_values_to_numpy_array(poses)
    return default_dict_list_values_to_numpy_array(poses)


def create_sequence_data_set_over_videos(data_set_path, frame_count):
    pass
