from itertools import cycle, islice
from statistics import mode

import numpy as np

from ivu.utils import (
    train_val_split,
    shuffle_two_list_together,
    one_hot,
    load_pickle,
)


class TrainInputData:
    def __init__(self, df, train_type, class_column="class_label_index"):
        self._df = df
        self._train_type = train_type
        self._class_column = class_column

    @staticmethod
    def _sequence(x, y, stride):
        _x_sequence = list()
        _y_sequence = list()

        n_samples = x.shape[0]
        indices = np.arange(n_samples)

        for start in range(0, n_samples, stride):
            end = min(start + stride, n_samples)

            batch_idx = indices[start:end]
            if len(batch_idx) < stride:
                batch_idx = list(islice(cycle(indices), stride))

            _x_sequence.append(x[batch_idx].reshape(stride, -1))
            _y_sequence.append(mode(y[batch_idx]))

        return _x_sequence, _y_sequence

    def create_sequence_data(
        self, validation_split: float = None, stride: int = 300, n_classes: int = 1
    ):
        files = np.unique(self._df["file"].to_numpy())

        _x_train = list()
        _y_train = list()

        for file in files:
            file_df = self._df.loc[self._df["file"] == file]

            x = np.array(file_df[self._train_type].tolist())
            y = np.array(file_df[self._class_column].tolist())

            _x, _y = self._sequence(x, y, stride)

            _x_train.extend(_x)
            _y_train.extend(_y)

        if validation_split is not None:
            _x_train, _y_train, _x_val, _y_val = train_val_split(
                np.array(_x_train),
                np.array(_y_train),
                val_split=validation_split,
                n_classes=n_classes,
            )
        else:
            _x_train, _y_train, _x_val, _y_val = (
                np.array(_x_train),
                np.array(_y_train),
                None,
                None,
            )
        return (
            _x_train,
            one_hot(_y_train, n_classes),
        ), None if _x_val is None or _y_val is None else (
            _x_val,
            one_hot(_y_val, n_classes),
        )

    @classmethod
    def data_with_normalized_key_points(cls, pth):
        df = load_pickle(pth)
        return cls(df, "normalized_key_points")

    @classmethod
    def data_with_normalized_distance_matrix(cls, pth):
        df = load_pickle(pth)
        return cls(df, "normalized_distance_matrix")

    @classmethod
    def data_with_distance_matrix(cls, pth):
        df = load_pickle(pth)
        return cls(df, "distance_matrix")

    @classmethod
    def data_with_key_points(cls, pth):
        pass
