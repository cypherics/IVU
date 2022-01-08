import os
import numpy as np

from py_oneliner import one_liner

from ivu.utils import (
    read_video,
    get_normalized_distance_matrix,
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

    @staticmethod
    def _adjust_frame_for_video_frame(x, stride):
        n_samples = x.shape[0]

        indices = np.arange(n_samples)
        end = min(0 + stride, n_samples)
        stride_idx = indices[0 - stride + end - 0 : end]
        return x[stride_idx]

    def data_for_normalized_distance_matrix(self):

        files = os.listdir(self._data_dir)
        for iterator, file in enumerate(files):
            input_data = list()
            file_path = os.path.join(*[self._data_dir, file])
            vr = read_video(
                file_path, width=self._frame_width, height=self._frame_height
            )

            for stride_iterator, frame in enumerate(range(len(vr))):
                one_liner.one_line(
                    tag=f"PROGRESS [VIDEOS: {iterator + 1}/{len(files)}] [CURRENT FILE : {file}]",
                    tag_data=f"[FRAMES : {frame + 1}/{len(vr)}]",
                    to_reset_data=True,
                    tag_color="red",
                    tag_data_color="red",
                )
                normalized_distance_matrix = get_normalized_distance_matrix(
                    pose_estimator=self._pose_estimator, rgb_input=vr[frame].asnumpy()
                )
                input_data.append(
                    normalized_distance_matrix[
                        np.triu_indices(normalized_distance_matrix.shape[0], k=1)
                    ]
                )
                if (stride_iterator + 1) % self._stride == 0:
                    yield iterator, file, frame, np.array(input_data)
                elif (
                    stride_iterator + 1 == len(vr)
                    and (stride_iterator + 1) % self._stride != 0
                ):
                    yield iterator, file, frame, self._adjust_frame_for_video_frame(
                        np.array(input_data), self._stride
                    )