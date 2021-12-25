import numpy as np
from abc import ABC, abstractmethod


class AbstractPoseEstimator(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_key_points_from_image(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_key_points_from_video(self, **kwargs) -> np.ndarray:
        pass


class PoseEstimationFailedError(Exception):
    pass
