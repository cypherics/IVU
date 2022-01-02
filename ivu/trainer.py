from ivu.conf import TrainConf
from ivu.dataset import TrainInputData


class Trainer:
    def __init__(self, train_data, val_data, config):
        self._train_data = train_data
        self._val_data = val_data
        self._config = config

    def start_training(self):
        model = self._config.get_model()
        model.compile(
            optimizer=self._config.get_optimizer(),
            loss=self._config.get_loss(),
            metrics="accuracy",
        )

        _ = model.fit(
            self._train_data[0],
            self._train_data[1],
            epochs=self._config.get_entry(name="epochs"),
            callbacks=self._config.get_callbacks(),
            batch_size=self._config.get_entry(name="batch_size"),
            validation_data=self._val_data,
        )

        print(f"TRAIN METRIC : {model.evaluate(*self._train_data)}")
        if self._val_data is not None:
            print(f"VAL METRIC : {model.evaluate(*self._val_data)}")

    @classmethod
    def train_with_normalized_key_points(cls):
        raise NotImplementedError

    @classmethod
    def train_with_normalized_distance_matrix(cls, data_pth, conf_pth):
        input_data = TrainInputData.data_with_normalized_distance_matrix(data_pth)
        conf = TrainConf(conf_pth)

        parameters = conf.get_entry("data")
        train_data, val_data = input_data.create_sequence_data(
            stride=parameters["stride"], validation_split=parameters["validation_split"]
        )
        return cls(train_data, val_data, conf)

    @classmethod
    def train_with_distance_matrix(cls):
        raise NotImplementedError

    @classmethod
    def train_with_key_points(cls):
        raise NotImplementedError
