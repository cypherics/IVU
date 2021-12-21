import pickle
import os
from collections import defaultdict

from typing import List
import numpy as np

from py_oneliner import one_liner


def sequence_generator(input_stream: np.ndarray, sequence_count: int):
    n_samples = input_stream.shape[0]

    indices = np.arange(n_samples)

    for start in range(0, n_samples, sequence_count):
        end = min(start + sequence_count, n_samples)

        batch_idx = indices[start:end]
        if len(batch_idx) < sequence_count:
            batch_idx = indices[start - sequence_count + end - start : end]
        yield input_stream[batch_idx]


def dataset_file_generator(data_set_path: str, classes: List):
    for iterator, class_label in enumerate(classes):
        files = os.listdir(os.path.join(data_set_path, class_label))
        for file_iterator, file in enumerate(files):
            one_liner.one_line(
                tag="PROGRESS",
                tag_data=f"CURRENT FILE : {file}, LABEL: {class_label}, : COUNTER : {file_iterator + 1}/{len(files)}",
                to_reset_data=True,
                tag_color="red",
                tag_data_color="yellow",
            )
            yield os.path.join(*[data_set_path, class_label, file]), iterator


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_pickle(obj, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def list_to_numpy(input_list: list) -> np.ndarray:
    return np.array(input_list)


def default_dict_list_values_to_numpy_array(
    input_dict: defaultdict[List],
) -> defaultdict:
    output = defaultdict()
    for key, value in input_dict.items():
        assert type(value) == list
        output[key] = list_to_numpy(value)
    return output
