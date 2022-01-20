import os
from dataclasses import dataclass

import numpy as np
import tensorflow
from py_oneliner import one_liner

from ivu.conf import DataConf
from ivu.data.infer import VideoInferenceInputData
from ivu.utils import read_video, write_to_video

LABEL_ASSOCIATION = {
    0: "back round",
    1: "back warp",
    2: "bad head",
    3: "inner thigh",
    4: "shallow",
    5: "bad toe",
    6: "good",
}


@dataclass
class FileInference:
    fps: int
    dim: tuple
    annotated_frames: list
    prediction_per_frame: list


class Inference:
    def __init__(
        self,
        model: tensorflow.keras.Model,
        video_parameters: dict,
        pose_estimator_parameters: dict,
        inference_parameters: dict,
        data_dir: str,
        save_dir: str = None,
    ):
        self._model = model

        self._video_parameters = video_parameters
        self._pose_estimator_parameters = pose_estimator_parameters
        self._inference_parameters = inference_parameters
        self._save_dir = save_dir
        self._data_dir = data_dir

        self._video_inference = self._init_video_inference()

    def _init_video_inference(self):
        video_inference = VideoInferenceInputData(
            data_dir=self._data_dir,
            **{
                **self._pose_estimator_parameters,
                **self._video_parameters,
                **self._inference_parameters,
            },
        )
        return video_inference

    @staticmethod
    def _load_model(pth):
        model = tensorflow.keras.models.load_model(pth, compile=True)
        return model

    def _infer_video(self, video_reader, video_inference):
        my_frames_collection = list()
        predictions_over_frame = list()
        video_stride = self._video_parameters["stride"]
        for (
            my_frames,
            input_data,
        ) in video_inference.inference_data_gen(video_reader=video_reader):
            input_data = np.expand_dims(input_data, axis=0)
            prediction = self._model.predict(input_data)
            predictions_over_frame.extend(
                [LABEL_ASSOCIATION[int(np.argmax(prediction))]] * video_stride
            )
            my_frames_collection.extend(my_frames)
        return my_frames_collection, predictions_over_frame

    def run_over_file(self, pth):
        vr = read_video(
            pth,
            width=self._video_parameters["frame_width"],
            height=self._video_parameters["frame_height"],
        )
        _, height, width, _ = vr[:].shape
        fps = int(vr.get_avg_fps())

        my_frames_collection, predictions = self._infer_video(
            video_reader=vr, video_inference=self._video_inference
        )
        return FileInference(
            fps=fps,
            dim=(height, width),
            annotated_frames=my_frames_collection,
            prediction_per_frame=predictions,
        )

    def run(self):

        files = os.listdir(self._data_dir)
        for iterator, file in enumerate(files):
            file_path = os.path.join(*[self._data_dir, file])
            one_liner.one_line(
                tag=f"PROGRESS",
                tag_data=f"[VIDEOS: {iterator + 1}/{len(files)}] [CURRENT FILE : {file}]",
                to_reset_data=True,
                tag_color="red",
                tag_data_color="red",
            )

            file_inference = self.run_over_file(file_path)

            if self._save_dir is not None:
                write_to_video(
                    os.path.join(
                        self._save_dir, f"output_{os.path.splitext(file)[0]}.mp4"
                    ),
                    image_sequence=file_inference.annotated_frames,
                    text_per_frame=file_inference.prediction_per_frame,
                    fps=file_inference.fps,
                    width=file_inference.dim[-1],
                    height=file_inference.dim[0],
                )

    @classmethod
    def init_inference_from_config_pth(cls, pth):
        conf = DataConf(pth)
        return cls(
            cls._load_model(conf.get_saved_model_pth()),
            video_parameters=conf.get_video_parameters(),
            pose_estimator_parameters=conf.get_pose_estimators_parameters(),
            inference_parameters=conf.get_inference_parameters(),
            save_dir=conf.get_entry("save_dir"),
            data_dir=conf.get_entry("data_dir"),
        )

    @classmethod
    def init_inference_from_config(cls, conf: DataConf):
        return cls(
            cls._load_model(conf.get_saved_model_pth()),
            video_parameters=conf.get_video_parameters(),
            pose_estimator_parameters=conf.get_pose_estimators_parameters(),
            inference_parameters=conf.get_inference_parameters(),
            save_dir=conf.get_entry("save_dir"),
            data_dir=conf.get_entry("data_dir"),
        )

    @classmethod
    def init_with_parameters(
        cls,
        model_pth,
        video_parameters: dict,
        pose_estimator_parameters: dict,
        inference_parameters: dict,
        data_dir: str,
        save_dir: str = None,
    ):
        return cls(
            cls._load_model(model_pth),
            video_parameters=video_parameters,
            pose_estimator_parameters=pose_estimator_parameters,
            inference_parameters=inference_parameters,
            data_dir=data_dir,
            save_dir=save_dir,
        )
