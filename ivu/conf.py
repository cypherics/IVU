import datetime
import os

import tensorflow
from omegaconf import OmegaConf
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from ivu import models


class Config:
    def __init__(self, pth: str):
        self._config = OmegaConf.load(pth)

        self._optimizer = None
        self._callbacks = None
        self._loss_function = None

        self._log_dir = self._config.log_dir
        self._model_pth, self._graph_pth = self._create_log_dir()

    def get_entry(self, name: str):
        return self._config[name]

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

        return os.path.join(*[self._log_dir, date_time_stamp, "chk"]), os.path.join(
            *[self._log_dir, date_time_stamp, "graphs"]
        )


#
# conf = Config(r"config/normalized_sequence_distance_matrix.yaml")
# conf.get_callbacks()
# conf.get_entry("epochs")
