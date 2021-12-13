import os
import cv2
import numpy as np

import utils
from pose_estimation.AbstractPoseEstimator import PoseEstimationFailedError


def extractPoses(datasetPath, poseEstimator):
    labels = list(os.listdir(datasetPath))
    poses = dict()

    print('[INFO]: found %d classes' % len(labels))

    for label in labels:
        labelPath = os.path.join(datasetPath, label)
        fnames = list(os.listdir(labelPath))
        poses[label] = list()

        print('[INFO]: processing class %s...' % label)
        print('[INFO]: progress: 0 / %d' % len(fnames), end='\r')
        
        for i, fname in enumerate(fnames):
            fpath = os.path.join(labelPath, fname)
            image = cv2.imread(fpath)
            
            try:
                poses[label].append(poseEstimator.getKeypointsFromImage(image))
            except PoseEstimationFailedError:
                pass
            
            print('[INFO]: progress: %d / %d' % (i, len(fnames)), end='\r')

        poses[label] = np.array(poses[label], dtype=np.float32)
        print('[INFO]: progress: %d / %d' % (len(fnames), len(fnames)))

    return poses


def extractMediapipe(datasetPath, complexity=2):
    """Mediapipe provides 3D pose estimation
    """
    print('[INFO]: extracting poses from dataset using MediPpipe (complexity=%d)...' % complexity)
    from pose_estimation.PoseEstimatorMediaPipe import PoseEstimatorMediaPipe
    poseEstimator = PoseEstimatorMediaPipe(complexity=complexity)

    return extractPoses(datasetPath, poseEstimator)


def extractOpenPose(datasetPath):
    """OpenPose provides 3D pose estimation
    """
    print('[INFO]: extracting poses from dataset using OpenPose...')
    from pose_estimation.PoseEstimatorOpenPose import PoseEstimatorOpenPose
    poseEstimator = PoseEstimatorOpenPose('F:/Programs/openpose/')

    return extractPoses(datasetPath, poseEstimator)
    

def extractActionAI(datasetPath):
    """ActionAI provides 2D pose estimation
    """
    print('[INFO]: extracting poses from dataset using ActionAI...')
    from pose_estimation.PoseEstimatorActionAI import PoseEstimatorActionAI
    poseEstimator = PoseEstimatorActionAI('./models/existing/ActionAIpose.tflite')

    return extractPoses(datasetPath, poseEstimator)


def extractMovenet(datasetPath):
    """Movenet provides 2D pose estimation
    """
    print('[INFO]: extracting poses from dataset using Movenet...')
    from pose_estimation.PoseEstimatorMovenet import PoseEstimatorMovenet
    poseEstimator = PoseEstimatorMovenet('./models/existing/movenet_thunder.tflite')

    return extractPoses(datasetPath, poseEstimator)


def main(datasetsPath):
    dataset1Path = os.path.join(datasetsPath, 'dataset1/')
    dataset2Path = os.path.join(datasetsPath, 'dataset2/train')

    datasetPaths = [dataset1Path, dataset2Path]
    
    poseExtractors = {
        'mediapipe_c0': lambda x: extractMediapipe(x, complexity=0),
        'mediapipe_c1': lambda x: extractMediapipe(x, complexity=1),
        'mediapipe_c2': lambda x: extractMediapipe(x, complexity=2),
        'actionai': extractActionAI,
        'openpose': extractOpenPose,
        'movenet': extractMovenet,
    }

    for i, datasetPath in enumerate(datasetPaths):

        for poseExtractorName in poseExtractors:
            outputPath = 'datasets/pose_ds%d_%s.pkl' % (i+1, poseExtractorName)

            if not os.path.exists(outputPath):
                poseExtractorFunc = poseExtractors[poseExtractorName]
                
                poses = poseExtractorFunc(datasetPath)
                print('[INFO]: saving poses to: %s' % outputPath)
                utils.save_pickle(poses, outputPath)


if __name__ == '__main__':
    main('F:\Studium\IVU\KU\project\datasets')
