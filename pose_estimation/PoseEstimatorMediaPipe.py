import cv2
import numpy as np
import mediapipe as mp

from .AbstractPoseEstimator import AbstractPoseEstimator, PoseEstimationFailedError


class PoseEstimatorMediaPipe(AbstractPoseEstimator):
    def __init__(self, complexity=2, staticImageMode=False):
        self.staticImageMode = staticImageMode
        self.complexity = complexity
        self.__init()

    def __del__(self):
        self.__release()

    def __init(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=not self.staticImageMode,
            model_complexity=self.complexity,
            min_detection_confidence=0.5,
            smooth_landmarks=True
        )
        self.pose = self.pose.__enter__()

    def __release(self):
        try:
            self.pose.__exit__(None, None, None)
        except:
            pass

    # override
    def reset(self):
        if not self.staticImageMode:
            self.__release()
            self.__init()

    # override
    def getKeypointsFromImage(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(image)
        if not results.pose_landmarks:
            raise PoseEstimationFailedError()

        # visibility might be useful
        # landmarks = [(p.x, p.y, p.z, p.visibility) for p in results.pose_landmarks]
        keypoints = [(p.x, p.y, p.z) for p in results.pose_landmarks.landmark]
        return np.array(keypoints)
