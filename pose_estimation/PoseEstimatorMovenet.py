import os
import sys
import numpy as np

from .AbstractPoseEstimator import AbstractPoseEstimator


def loadModel(modelPath):
    sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), 'movenet')))

    from ml.movenet import Movenet
    return Movenet(modelPath)


def personToKeypoints(person):
    N_keypoints = len(person.keypoints)
    keypoints = np.empty((N_keypoints, 2), dtype=np.float32)

    for k in person.keypoints:
        keypoints[k.body_part.value] = (k.coordinate.x, k.coordinate.y)

    return keypoints


class PoseEstimatorMovenet(AbstractPoseEstimator):
    def __init__(self, modelPath, inferenceCount=2):
        self.model = loadModel(modelPath)
        self.inferenceCount = inferenceCount

    # override
    def reset(self):
        pass
    
    # override
    def getKeypointsFromImage(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        if image.dtype != np.float32:
            raise ValueError("Image must be of type float32 or uint8")

        person = self.model.detect(image, reset_crop_region=True)
        for _ in range(self.inferenceCount - 1):
            person = self.model.detect(image, reset_crop_region=False)
        
        return personToKeypoints(person)
