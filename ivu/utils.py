import pickle
import os
import random
from collections import defaultdict
from typing import List

import decord
import pandas as pd
import numpy as np
from decord import VideoReader, cpu


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
