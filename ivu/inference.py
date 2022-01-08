from collections import defaultdict

import numpy as np
import tensorflow
from ivu.conf import DataConf
from ivu.data.infer import VideoInferenceInputData


class Inference:
    def __init__(self, model: tensorflow.keras.Model, conf: DataConf):
        self._model = model
        self._conf = conf

    @staticmethod
    def _load_model(pth):
        model = tensorflow.keras.models.load_model(pth, compile=True)
        return model

    def run(self):

        pose_estimator_param = self._conf.get_pose_estimators_parameters()
        video_param = self._conf.get_video_parameters()
        inference_param = self._conf.get_inference_parameters()

        video_inference = VideoInferenceInputData(
            data_dir=self._conf.get_entry("data_dir"),
            **{**pose_estimator_param, **video_param, **inference_param}
        )

        predictions = defaultdict(list)
        for (
            file_iterator,
            file,
            frame,
            input_data,
        ) in video_inference.data_for_normalized_distance_matrix():
            input_data = np.expand_dims(input_data, axis=0)
            prediction = self._model.predict(input_data)
            predictions[file].append(np.argmax(prediction))
            print(np.argmax(prediction))

    @classmethod
    def init_inference_from_config(cls, pth):
        conf = DataConf(pth)
        return cls(cls._load_model(conf.get_saved_model_pth()), conf)
