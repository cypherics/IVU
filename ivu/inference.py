import os
from collections import defaultdict

import numpy as np
import skvideo.io
import tensorflow
from py_oneliner import one_liner

from ivu.conf import DataConf
from ivu.data.infer import VideoInferenceInputData
from ivu.pose_estimator.base_pose_estimator import PoseEstimationFailedError
from ivu.utils import read_video


class Inference:
    def __init__(self, model: tensorflow.keras.Model, conf: DataConf):
        self._model = model
        self._conf = conf

    @staticmethod
    def _load_model(pth):
        model = tensorflow.keras.models.load_model(pth, compile=True)
        return model

    def infer_video(self, video_reader, video_inference, file):
        predictions = defaultdict(list)
        my_frames_collection = list()
        for (
            my_frames,
            input_data,
        ) in video_inference.inference_data_gen(video_reader=video_reader):
            input_data = np.expand_dims(input_data, axis=0)
            prediction = self._model.predict(input_data)
            predictions[file].append(np.argmax(prediction))
            my_frames_collection.extend(my_frames)
        return my_frames_collection, predictions

    def run(self):

        pose_estimator_param = self._conf.get_pose_estimators_parameters()
        video_param = self._conf.get_video_parameters()
        inference_param = self._conf.get_inference_parameters()
        save_dir = self._conf.get_entry("save_dir")

        video_inference = VideoInferenceInputData(
            data_dir=self._conf.get_entry("data_dir"),
            **{**pose_estimator_param, **video_param, **inference_param},
        )

        files = os.listdir(self._conf.get_entry("data_dir"))
        for iterator, file in enumerate(files):
            file_path = os.path.join(*[self._conf.get_entry("data_dir"), file])
            vr = read_video(
                file_path,
                width=video_param["frame_width"],
                height=video_param["frame_height"],
            )
            one_liner.one_line(
                tag=f"PROGRESS",
                tag_data=f"[VIDEOS: {iterator + 1}/{len(files)}] [CURRENT FILE : {file}]",
                to_reset_data=True,
                tag_color="red",
                tag_data_color="red",
            )
            my_frames_collection, predictions = self.infer_video(
                video_reader=vr, video_inference=video_inference, file=file
            )

            skvideo.io.vwrite(
                os.path.join(save_dir, f"output_{file.split('.')[0]}.mp4"),
                np.array(my_frames_collection).astype(np.uint8),
            )

            print(predictions)

    @classmethod
    def init_inference_from_config(cls, pth):
        conf = DataConf(pth)
        return cls(cls._load_model(conf.get_saved_model_pth()), conf)
