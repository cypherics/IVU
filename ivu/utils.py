import pickle
import os
from collections import defaultdict

import decord
import pandas as pd
import numpy as np
from decord import VideoReader, cpu


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
