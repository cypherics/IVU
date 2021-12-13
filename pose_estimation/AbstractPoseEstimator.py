import numpy as np
from abc import ABC, abstractmethod


class AbstractPoseEstimator(ABC):
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def getKeypointsFromImage(self, image: np.ndarray) -> np.ndarray:
        pass


class PoseEstimationFailedError(Exception):
    pass
