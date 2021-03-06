import datetime
import os

import tensorflow
from omegaconf import OmegaConf
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from ivu import models
from ivu.utils import log_in_tmp_dir, log_in_current_dir


class Conf:
    def __init__(self, pth):
        self._config = OmegaConf.load(pth)
        if "log_dir" not in self._config.keys():
            log_dir = log_in_tmp_dir()
            self._config["log_dir"] = log_dir

        elif self._config["log_dir"] is None:
            log_dir = log_in_current_dir()
            self._config["log_dir"] = log_dir

    def get_entry(self, name: str):
        return self._config[name]

    def get_sub_value_entry(self, name: str, sub_value: str):
        return self._config[name][sub_value]

    def update_sub_entry(self, name: str, sub_name: str, value: str):
        self._config[name][sub_name] = value

    def update_entry(self, name: str, value: str):
        self._config[name] = value

    def contains_entry(self, name: str):
        return hasattr(self._config, name)

    def contains_sub_entry(self, name: str, sub_name: str):
        return hasattr(self._config[name], sub_name)


class TrainConf(Conf):
    def __init__(self, pth: str):
        super().__init__(pth)

        self._optimizer = None
        self._callbacks = None
        self._loss_function = None

        self._log_dir = self._config.log_dir
        self._model_pth, self._graph_pth, self._test_metric = self._create_log_dir()

    def get_loss(self):
        return getattr(losses, self._config.loss.name)(**self._config.loss.parameters)

    def get_optimizer(self):
        return getattr(optimizers, self._config.optimizer.name)(
            **self._config.optimizer.parameters
        )

    def _get_tensor_board_call_back(self):
        return callbacks.TensorBoard(log_dir=self._graph_pth, histogram_freq=1)

    def get_model(self) -> tensorflow.keras.Model:
        return getattr(models, self._config.model.name)(**self._config.model.parameters)

    def get_callbacks(self):
        callback_collection = list()
        for key, value in self._config.callbacks.items():
            if key == "ModelCheckpoint":
                value.parameters.filepath = self._model_pth
            callback_collection.append(getattr(callbacks, key)(**value.parameters))
        callback_collection.append(self._get_tensor_board_call_back())
        return callback_collection

    def _create_log_dir(self):
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

        date_time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        return (
            os.path.join(*[self._log_dir, date_time_stamp, "chk"]),
            os.path.join(*[self._log_dir, date_time_stamp, "graphs"]),
            os.path.join(*[self._log_dir, date_time_stamp]),
        )

    def get_test_metric_dir(self):
        return self._test_metric


class DataConf(Conf):
    def __init__(self, pth):
        super().__init__(pth)

    def get_saved_model_pth(self):
        return self.get_sub_value_entry("inference", "model_pth")

    def get_video_parameters(self):
        return self.get_entry("video")

    def get_pose_estimators_parameters(self):
        return self.get_entry("pose")

    def get_inference_parameters(self):
        return self.get_entry("inference")
