from statistics import mode
import numpy as np


from ivu.utils import (
    train_val_split,
    shuffle_two_list_together,
    one_hot,
    load_pickle,
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
