import os.path

import numpy as np
import pandas as pd
import tensorflow
import seaborn as sn
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ivu.conf import TrainConf
from ivu.data.testing import TestInputData
from ivu.data.training import TrainInputData

LABEL_ASSOCIATION = {
    0: "back round",
    1: "back warp",
    2: "bad head",
    3: "inner thigh",
    4: "shallow",
    5: "bad toe",
    6: "good",
}


class Metric:
    @staticmethod
    def model_metric(model, data):
        return model.evaluate(*data)

    @staticmethod
    def generate_confusion_matrix(model, data):
        predictions = tensorflow.argmax(model.predict(data[0]), axis=-1)
        confusion_matrix = tensorflow.math.confusion_matrix(
            tensorflow.argmax(data[1], axis=-1), predictions
        )
        return np.array(confusion_matrix)

    def generate_metric(self, model, data, save_dir, title, split="test"):
        with PdfPages(os.path.join(save_dir, f"{title}_{split.upper()}.pdf")) as pdf:
            fig = plt.figure(figsize=(11.69, 8.27))
            df_cm = pd.DataFrame(
                self.generate_confusion_matrix(model, data),
                LABEL_ASSOCIATION.values(),
                LABEL_ASSOCIATION.values(),
            )
            sn.set(font_scale=1)
            sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
            plt.plot()
            plt.title(title)

            metric = self.model_metric(model, data)

            txt = f"{split.upper()} ACCURACY : {metric[-1]}, {split.upper()} LOSS: {metric[0]}"
            plt.text(0.05, 0.95, txt, transform=fig.transFigure, size=10)
            pdf.savefig()
            plt.close()


class Trainer:
    def __init__(self, train_data, val_data, test_data, config, run_type: str = "RUN"):
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data

        self._config = config
        self._run_type = run_type

        self._metric = Metric()

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
            shuffle=self._config.get_entry("shuffle")
            if self._config.contains_entry("shuffle")
            else False,
        )

        print(f"TRAIN METRIC : {model.evaluate(*self._train_data)}")
        if self._val_data is not None:
            self._metric.generate_metric(
                model,
                self._val_data,
                self._config.get_test_metric_dir(),
                self._run_type,
                split="val",
            )
            print(f"VAL METRIC : {model.evaluate(*self._val_data)}")

        if self._test_data is not None:
            self._metric.generate_metric(
                model,
                self._test_data,
                self._config.get_test_metric_dir(),
                self._run_type,
            )

    @classmethod
    def train_with_normalized_key_points(cls, train_pth, test_pth, conf_pth):
        input_train_data = TrainInputData.data_with_normalized_key_points(train_pth)

        conf = TrainConf(conf_pth)

        parameters = conf.get_entry("data")
        train_data, val_data = input_train_data.create_sequence_data(
            stride=parameters["stride"],
            validation_split=parameters["validation_split"],
            n_classes=conf.get_sub_value_entry("model", "parameters")["n_classes"],
        )

        if test_pth is not None:
            input_test_data = TestInputData.data_with_normalized_key_points(test_pth)
            test_data, _ = input_test_data.create_sequence_data(
                stride=parameters["stride"],
                validation_split=None,
                n_classes=conf.get_sub_value_entry("model", "parameters")["n_classes"],
            )
        else:
            test_data = None

        return cls(
            train_data,
            val_data,
            test_data,
            conf,
            run_type=f"{conf.get_sub_value_entry('model', 'name')} - NORMALIZED KEY POINTS",
        )

    @classmethod
    def train_with_normalized_distance_matrix(cls, train_pth, test_pth, conf_pth):
        input_train_data = TrainInputData.data_with_normalized_distance_matrix(
            train_pth
        )
        conf = TrainConf(conf_pth)

        parameters = conf.get_entry("data")
        train_data, val_data = input_train_data.create_sequence_data(
            stride=parameters["stride"],
            validation_split=parameters["validation_split"],
            n_classes=conf.get_sub_value_entry("model", "parameters")["n_classes"],
        )
        if test_pth is not None:
            input_test_data = TestInputData.data_with_normalized_distance_matrix(test_pth)

            test_data, _ = input_test_data.create_sequence_data(
                stride=parameters["stride"],
                validation_split=None,
                n_classes=conf.get_sub_value_entry("model", "parameters")["n_classes"],
            )
        else:
            test_data = None
        return cls(
            train_data,
            val_data,
            test_data,
            conf,
            run_type=f"{conf.get_sub_value_entry('model', 'name')} - NORMALIZED DISTANCE MATRIX",
        )

    @classmethod
    def train_with_distance_matrix(cls, train_pth, test_pth, conf_pth):
        input_train_data = TrainInputData.data_with_distance_matrix(train_pth)

        conf = TrainConf(conf_pth)

        parameters = conf.get_entry("data")
        train_data, val_data = input_train_data.create_sequence_data(
            stride=parameters["stride"],
            validation_split=parameters["validation_split"],
            n_classes=conf.get_sub_value_entry("model", "parameters")["n_classes"],
        )

        if test_pth is not None:
            input_test_data = TestInputData.data_with_distance_matrix(test_pth)
            test_data, _ = input_test_data.create_sequence_data(
                stride=parameters["stride"],
                validation_split=None,
                n_classes=conf.get_sub_value_entry("model", "parameters")["n_classes"],
            )
        else:
            test_data = None

        return cls(
            train_data,
            val_data,
            test_data,
            conf,
            run_type=f"{conf.get_sub_value_entry('model', 'name')} - DISTANCE MATRIX",
        )

    @classmethod
    def train_with_key_points(cls):
        raise NotImplementedError
