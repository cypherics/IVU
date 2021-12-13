import os
import sys
import importlib
import numpy as np

from .AbstractPoseEstimator import AbstractPoseEstimator


def importOpenPose(openPosePath):
    # e.g. openPosePath = 'F:/Programs/openpose/'
    sys.path.append(os.path.join(openPosePath, 'bin/python/openpose/Release'))
    os.environ['PATH'] += ';' + os.path.join(openPosePath, 'bin/python/openpose/Release')
    os.environ['PATH'] += ';' + os.path.join(openPosePath, 'bin')
    return importlib.import_module('pyopenpose')


class PoseEstimatorOpenPose(AbstractPoseEstimator):
    def __init__(self, openPosePath):
        self.op = importOpenPose(openPosePath)
        
        params = {
            'model_folder': os.path.join(openPosePath, 'models'),
        }
        self.opWrapper = self.op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    # override
    def reset(self):
        pass
    
    # override
    def getKeypointsFromImage(self, image: np.ndarray) -> np.ndarray:
        datum = self.op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))
        keypoints = np.array(datum.poseKeypoints)[0]
        return keypoints
