from collections import defaultdict
from itertools import islice, cycle

import decord
import numpy as np

from py_oneliner import one_liner

from ivu.pose_estimator.base_pose_estimator import PoseEstimationFailedError
from ivu.pose_estimator.pose_landmarks import Pose16LandmarksBodyModel
from ivu.utils import (
    get_distance_matrix_from_key_points,
    get_normalized_distance_matrix_from_body_key_points,
    normalize_body_key_points,
)
from ivu.pose_estimator.media_pipe_estimator import get_media_pipe_pose_estimator


class VideoInferenceInputData:
    def __init__(
        self,
        data_dir=None,
        pose_estimator_complexity=1,
        use_pose_estimator_over_static_image=True,
        frame_width=-1,
        frame_height=-1,
        stride=128,
        infer_for="normalized_distance_matrix",
        **kwargs,
    ):
        self._pose_estimator = get_media_pipe_pose_estimator(
            complexity=pose_estimator_complexity,
            static_image_mode=use_pose_estimator_over_static_image,
        )
        self._data_dir = data_dir
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._stride = stride
        self._infer_for = self._inference_function_dispatcher()[infer_for]

        self._meta = None

    def _inference_function_dispatcher(self):
        return {
            "normalized_distance_matrix": self._get_inference_normalized_distance_matrix,
            "distance_matrix": self._get_inference_distance_matrix,
            "normalized_key_points": self._get_inference_body_normalized_key_points,
        }

    def _get_inference_distance_matrix(self, rgb_input):
        key_points = self._pose_estimator.get_key_points_from_image(rgb_input)
        dist_mat = get_distance_matrix_from_key_points(key_points)
        return key_points, dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]

    def _get_inference_normalized_distance_matrix(self, rgb_input):
        key_points = self._pose_estimator.get_key_points_from_image(rgb_input)
        dist_mat = get_normalized_distance_matrix_from_body_key_points(key_points)
        return key_points, dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]

    def _get_inference_body_normalized_key_points(self, rgb_input):
        key_points = self._pose_estimator.get_key_points_from_image(rgb_input)
        return key_points, normalize_body_key_points(
            key_points, Pose16LandmarksBodyModel
        )

    def _reset_meta(self):
        self._meta = defaultdict(list)

    @staticmethod
    def _adjust_frame_for_video_frame(x, stride):
        return np.array(list(islice(cycle(x), stride)))

    def inference_data_gen(self, video_reader: decord.VideoReader):
        self._reset_meta()
        for stride_iterator, frame_idx in enumerate(range(len(video_reader))):
            one_liner.one_line(
                tag="[FRAMES",
                tag_data=f"{frame_idx + 1}/{len(video_reader)}]",
                tag_color="red",
                tag_data_color="red",
            )
            frame = video_reader[frame_idx].asnumpy()
            try:
                key_points, pose_data = self._infer_for(
                    rgb_input=frame,
                )
                self._meta["pose_data"].append(pose_data)
                self._meta["my_frames"].append(
                    self._pose_estimator.get_annotated_frame()
                )

                if (stride_iterator + 1) % self._stride == 0 and len(
                    np.array(self._meta["pose_data"])
                ) == self._stride:
                    yield self._meta["my_frames"], np.array(self._meta["pose_data"])
                    self._reset_meta()
                elif (
                    stride_iterator + 1 == len(video_reader)
                    and (stride_iterator + 1) % self._stride != 0
                ):
                    yield self._meta["my_frames"], self._adjust_frame_for_video_frame(
                        np.array(self._meta["pose_data"]), self._stride
                    )
                    self._reset_meta()
                elif (
                    len(np.array(self._meta["pose_data"])) != self._stride
                    and (stride_iterator + 1) % self._stride == 0
                ):
                    yield self._meta["my_frames"], self._adjust_frame_for_video_frame(
                        np.array(self._meta["pose_data"]), self._stride
                    )
                    self._reset_meta()

            except PoseEstimationFailedError:
                pass
