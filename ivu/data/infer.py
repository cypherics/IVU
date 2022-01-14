from collections import OrderedDict

import decord
import numpy as np

from py_oneliner import one_liner

from ivu.pose_estimator.base_pose_estimator import PoseEstimationFailedError
from ivu.utils import (
    inference_function_dispatcher,
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
        self._infer_for = inference_function_dispatcher()[infer_for]

    @staticmethod
    def _adjust_frame_for_video_frame(x, stride):
        n_samples = x.shape[0]

        indices = np.arange(n_samples)
        end = min(0 + stride, n_samples)
        stride_idx = indices[0 - stride + end - 0 : end]
        return x[stride_idx]

    def inference_data_gen(self, video_reader: decord.VideoReader):
        input_data = list()
        my_frames = list()

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
                    pose_estimator=self._pose_estimator,
                    rgb_input=frame,
                )
                input_data.append(pose_data)
                my_frames.append(self._pose_estimator.get_annotated_frame())

                if (stride_iterator + 1) % self._stride == 0:
                    yield my_frames, np.array(input_data)
                elif (
                    stride_iterator + 1 == len(video_reader)
                    and (stride_iterator + 1) % self._stride != 0
                ):
                    yield my_frames, self._adjust_frame_for_video_frame(
                        np.array(input_data), self._stride
                    )
            except PoseEstimationFailedError:
                pass
