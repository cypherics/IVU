import os
from statistics import mode
from collections import defaultdict

import cv2
import numpy as np

from py_oneliner import one_liner

from ivu.pose_estimator.base_pose_estimator import PoseEstimationFailedError
from ivu.pose_estimator.media_pipe_estimator import get_media_pipe_pose_estimator


from ivu.utils import (
    folder_generator,
    save_pickle,
    read_video,
    train_val_split,
    shuffle_two_list_together,
    one_hot,
    load_pickle,
    get_pose_data_from_rgb_frame,
    get_normalized_distance_matrix,
)


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
                if stride_iterator + 1 % self._stride == 0:
                    yield iterator, file, np.array(input_data)
                elif (
                    stride_iterator + 1 == len(vr)
                    and stride_iterator + 1 % self._stride != 0
                ):
                    yield iterator, file, self._adjust_frame_for_video_frame(
                        np.array(input_data), self._stride
                    )


class TrainInputData:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def create_sequence_data(self, validation_split: float = None, stride: int = 300):
        _x = list()
        _y = list()

        n_samples = self._x.shape[0]
        indices = np.arange(n_samples)

        for start in range(0, n_samples, stride):
            end = min(start + stride, n_samples)

            batch_idx = indices[start:end]
            if len(batch_idx) < stride:
                batch_idx = indices[start - stride + end - start : end]

            _x.append(self._x[batch_idx])
            _y.append(mode(self._y[batch_idx]))
            # _y.append(Counter(y[batch_idx]).most_common(1)[0][0])

        _x_shuffled, _y_shuffled = shuffle_two_list_together(_x, _y)
        if validation_split is not None:
            _x_train, _y_train, _x_val, _y_val = train_val_split(
                np.array(_x_shuffled), np.array(_y_shuffled)
            )
        else:
            _x_train, _y_train, _x_val, _y_val = (
                np.array(_x_shuffled),
                np.array(_y_shuffled),
                None,
                None,
            )
        return (
            _x_train,
            one_hot(_y_train),
        ), None if _x_val is None or _y_val is None else (_x_val, one_hot(_y_val))

    @classmethod
    def data_with_normalized_key_points(cls, pth):
        pass

    @classmethod
    def data_with_normalized_distance_matrix(cls, pth):
        df = load_pickle(pth)
        subset = df[
            [
                "normalized_distance_matrix",
                "class_label",
                "class_label_index",
                "frame_details",
                "frame_number",
            ]
        ]
        x = np.array(subset["normalized_distance_matrix"].tolist())
        y = np.array(subset["class_label_index"].tolist())

        return cls(x, y)

    @classmethod
    def data_with_distance_matrix(cls, pth):
        pass

    @classmethod
    def data_with_key_points(cls, pth):
        pass


class GeneratedData:
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


def generate_training_data_over_images(
    data_set_dir: str,
    pose_estimator_complexity=1,
    use_pose_estimator_over_static_image=True,
) -> GeneratedData:
    pose_estimator = get_media_pipe_pose_estimator(
        complexity=pose_estimator_complexity,
        static_image_mode=use_pose_estimator_over_static_image,
    )

    data = GeneratedData()
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


def generate_training_data_over_videos(
    data_set_dir: str,
    pose_estimator_complexity=1,
    use_pose_estimator_over_static_image=True,
    frame_width=-1,
    frame_height=-1,
) -> GeneratedData:
    pose_estimator = get_media_pipe_pose_estimator(
        complexity=pose_estimator_complexity,
        static_image_mode=use_pose_estimator_over_static_image,
    )

    data = GeneratedData()
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


def training_data_set_over_images(
    data_set_dir: str,
    save_dir: str,
    pose_estimator_complexity=1,
    use_pose_estimator_over_static_image=True,
):
    data = generate_training_data_over_images(
        data_set_dir=data_set_dir,
        pose_estimator_complexity=pose_estimator_complexity,
        use_pose_estimator_over_static_image=use_pose_estimator_over_static_image,
    )
    data.store(save_dir=save_dir)


def training_data_set_over_videos(
    data_set_dir: str,
    save_dir: str,
    pose_estimator_complexity=1,
    use_pose_estimator_over_static_image=True,
    width=-1,
    height=-1,
):
    data = generate_training_data_over_videos(
        data_set_dir=data_set_dir,
        pose_estimator_complexity=pose_estimator_complexity,
        use_pose_estimator_over_static_image=use_pose_estimator_over_static_image,
        frame_width=width,
        frame_height=height,
    )
    data.store(save_dir=save_dir)
