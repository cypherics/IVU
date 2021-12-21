import cv2
import numpy as np
import mediapipe as mp

from ivu.pose_estimator.base_pose_estimator import (
    AbstractPoseEstimator,
    PoseEstimationFailedError,
)
from ivu.pose_estimator.pose_landmarks import (
    Pose16LandmarksBodyModel,
    MediaPipePoseLandmarks,
)


class PoseEstimatorMediaPipe(AbstractPoseEstimator):
    def __init__(self, complexity=2, static_image_mode=False, use_16_key_points=True):
        self.static_image_mode = static_image_mode
        self.complexity = complexity
        self.use_16_key_points = use_16_key_points
        self.__compute_body_model_16_key_points()

        self.__init()

    def __del__(self):
        self.__release()

    def __init(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=not self.static_image_mode,
            model_complexity=self.complexity,
            min_detection_confidence=0.5,
            smooth_landmarks=True,
        )
        self.pose = self.pose.__enter__()

    def __release(self):
        try:
            self.pose.__exit__(None, None, None)
        except Exception as ex:
            pass

    # override
    def reset(self):
        if not self.static_image_mode:
            self.__release()
            self.__init()

    # override
    def get_key_points_from_image(self, image: np.ndarray) -> np.ndarray:
        results = self.pose.process(image)
        if not results.pose_landmarks:
            raise PoseEstimationFailedError()

        # visibility might be useful
        # landmarks = [(p.x, p.y, p.z, p.visibility) for p in results.pose_landmarks]
        key_points = [(p.x, p.y, p.z) for p in results.pose_landmarks.landmark]
        return (
            np.array(key_points)[:, self._model_16_key_points]
            if self.use_16_key_points
            else np.array(key_points)
        )

    def get_key_points_from_video(self, **kwargs) -> np.ndarray:
        pass

    def __compute_body_model_16_key_points(self):
        names = [x.name for x in Pose16LandmarksBodyModel]
        names_media_pipe = [x.name for x in MediaPipePoseLandmarks]
        self._model_16_key_points = np.array([names_media_pipe.index(x) for x in names])


def get_media_pipe_pose_estimator(
    complexity: int = 2, static_image_mode: bool = False, use_16_key_points: bool = True
):
    return PoseEstimatorMediaPipe(
        complexity=complexity,
        static_image_mode=static_image_mode,
        use_16_key_points=use_16_key_points,
    )
