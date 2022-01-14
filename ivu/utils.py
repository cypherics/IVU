import os
import random
import tempfile
from collections import defaultdict
from typing import List

import decord
import pandas as pd
import numpy as np
from decord import VideoReader, cpu
from scipy.spatial.distance import squareform, pdist

from ivu.pose_estimator.pose_landmarks import (
    landmarks_to_embedding,
    Pose16LandmarksBodyModel,
)


def log_in_tmp_dir():
    # https://stackoverflow.com/questions/847850/cross-platform-way-of-getting-temp-directory-in-python
    log_dir = os.path.join(tempfile.gettempdir(), "ivu_logs")
    if not os.path.exists:
        os.makedirs(log_dir)
    return log_dir


def log_in_current_dir():
    log_dir = os.path.join(os.getcwd(), "ivu_logs")
    if not os.path.exists:
        os.makedirs(log_dir)
    return log_dir


def get_class_association(df):
    class_ass = dict()
    for i in range(0, 7):
        d = filter_data_frame(df, "class_label_index", i)["class_label"]
        class_ass[i] = np.unique(d.to_numpy())[0]
    return class_ass


def filter_data_frame(df: pd.DataFrame, name, value):
    return df.loc[df[name] == value]


def shuffle_two_list_together(x, y):
    # https://stackoverflow.com/a/23289591
    c = list(zip(x, y))
    random.shuffle(c)
    return zip(*c)


def train_val_split(x_train: np.ndarray, y_train: np.ndarray, val_split=0.2):
    # VAL SPLIT
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    val_data_count = int(x_train.shape[0] * val_split)

    x_val = x_train[:val_data_count]
    y_val = y_train[:val_data_count]

    x_train = np.delete(x_train, indices[:val_data_count], axis=0)
    y_train = np.delete(y_train, indices[:val_data_count])
    return x_train, y_train, x_val, y_val


def one_hot(input_data):
    # One hot encoding of the network outputs
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    input_data = input_data.reshape(len(input_data))
    n_values = int(np.max(input_data)) + 1
    return np.eye(n_values)[np.array(input_data, dtype=np.int32)]


def sequence_generator(input_stream: np.ndarray, sequence_count: int):
    n_samples = input_stream.shape[0]

    indices = np.arange(n_samples)

    for start in range(0, n_samples, sequence_count):
        end = min(start + sequence_count, n_samples)

        batch_idx = indices[start:end]
        if len(batch_idx) < sequence_count:
            batch_idx = indices[start - sequence_count + end - start : end]
        yield input_stream[batch_idx]


def folder_generator(input_path: str):
    sub_folder_collection = os.listdir(input_path)
    for sub_folder_index, sub_folder in enumerate(sub_folder_collection):
        yield sub_folder_index, sub_folder


def load_pickle(file_path) -> pd.DataFrame:
    return pd.read_pickle(file_path)


def save_pickle(obj, file_path):
    df = pd.DataFrame(obj)
    df.to_pickle(file_path)


def list_to_numpy(input_list: list) -> np.ndarray:
    return np.array(input_list)


def default_dict_list_values_to_numpy_array(
    input_dict: defaultdict,
) -> defaultdict:
    output = defaultdict()
    for key, value in input_dict.items():
        assert type(value) == list
        output[key] = list_to_numpy(value)
    return output


def read_video(pth, width=-1, height=-1) -> decord.VideoReader:
    return VideoReader(pth, ctx=cpu(0), width=width, height=height)


def extract_from_data_frames(data_frame: pd.DataFrame, columns: List):
    return data_frame[columns]


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


def normalize_body_key_points(body_key_points, body_model):
    return landmarks_to_embedding(body_key_points, body_model)[0]


def get_body_key_points(pose_estimator, rgb_input):
    return pose_estimator.get_key_points_from_image(rgb_input)


def get_distance_matrix(pose_estimator, rgb_input):
    body_key_points = pose_estimator.get_key_points_from_image(rgb_input)
    return get_distance_matrix_from_key_points(np.array(body_key_points))


def get_distance_matrix_from_key_points(body_key_points):
    return squareform(pdist(np.array(body_key_points)))


def get_body_normalized_key_points(pose_estimator, rgb_input):
    key_points = pose_estimator.get_key_points_from_image(rgb_input)
    return key_points, normalize_body_key_points(key_points, Pose16LandmarksBodyModel)


def get_normalized_distance_matrix(pose_estimator, rgb_input):
    body_key_points = get_body_normalized_key_points(pose_estimator, rgb_input)
    return get_distance_matrix_from_key_points(body_key_points)


def get_normalized_distance_matrix_from_body_key_points(body_key_points):
    return squareform(
        pdist(normalize_body_key_points(body_key_points, Pose16LandmarksBodyModel))
    )


def get_inference_distance_matrix(pose_estimator, rgb_input):
    key_points = pose_estimator.get_key_points_from_image(rgb_input)
    dist_mat = get_distance_matrix_from_key_points(key_points)

    return key_points, dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]


def get_inference_normalized_distance_matrix(pose_estimator, rgb_input):
    key_points = pose_estimator.get_key_points_from_image(rgb_input)
    dist_mat = get_normalized_distance_matrix_from_body_key_points(key_points)
    return key_points, dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]


def inference_function_dispatcher():
    return {
        "normalized_distance_matrix": get_inference_normalized_distance_matrix,
        "distance_matrix": get_inference_distance_matrix,
        "normalized_key_points": get_body_normalized_key_points,
    }
