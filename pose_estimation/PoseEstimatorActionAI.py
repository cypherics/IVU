import cv2
import numpy as np
import tensorflow as tf

from .AbstractPoseEstimator import AbstractPoseEstimator


class PoseEstimatorActionAI(AbstractPoseEstimator):
    def __init__(self, modelPath):
        self.model = tf.lite.Interpreter(model_path=modelPath)
        self.model.allocate_tensors()
        
        self.inputDetails = self.model.get_input_details()
        self.outputDetails = self.model.get_output_details()
        _, self.mpDim, _, self.numKeyPoints = self.outputDetails[0]['shape']
        self.imageInputSize = (self.inputDetails[0]['shape'][1], self.inputDetails[0]['shape'][2])

    # override
    def reset(self):
        pass
    
    # override
    def getKeypointsFromImage(self, image: np.ndarray) -> np.ndarray:
        # resize image for model
        image = cv2.resize(image, self.imageInputSize, interpolation=cv2.INTER_NEAREST)
        image = np.expand_dims(image.astype(self.inputDetails[0]['dtype'])[:, :, :3], axis=0)
        
        # run model
        self.model.set_tensor(self.inputDetails[0]['index'], image)
        self.model.invoke()
        result = self.model.get_tensor(self.outputDetails[0]['index'])
        
        res = result.reshape(1, self.mpDim**2, self.numKeyPoints)
        coords = np.divmod(np.argmax(res, axis=1), self.mpDim)
        
        return np.vstack(coords).T
