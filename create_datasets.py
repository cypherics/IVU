import os
import numpy as np

import utils
import preprocessing
from pose_estimation.pose_landmarks import PoseLandmarks, MediaPipePoseLandmarks


def postprocess(poses, testSplit):
    X_train, Y_train = list(), list()
    X_test, Y_test = list(), list()
    labels = list()

    for i, label in enumerate(poses):
        X = poses[label]
        Y = np.full(X.shape[0], i, dtype=np.float32)

        split = int(X.shape[0] * testSplit)
        X_test.append(X[:split])
        Y_test.append(Y[:split])
        X_train.append(X[split:])
        Y_train.append(Y[split:])
        
        labels.append(label)

    dataset = {
        'labels': labels,
        'X_test': np.concatenate(X_test, axis=0),
        'Y_test': np.concatenate(Y_test, axis=0),
        'X_train': np.concatenate(X_train, axis=0),
        'Y_train': np.concatenate(Y_train, axis=0),
    }

    return dataset


def selectKeypointsMediapipe(dataset):
    names = [x.name for x in PoseLandmarks]
    namesMP = [x.name for x in MediaPipePoseLandmarks]
    idx = np.array([namesMP.index(x) for x in names])

    return {
        'labels': dataset['labels'],
        'X_test': dataset['X_test'][:, idx],
        'Y_test': dataset['Y_test'],
        'X_train': dataset['X_train'][:, idx],
        'Y_train': dataset['Y_train'],
    }


def normalizeDataset(dataset):
    return {
        'labels': dataset['labels'],
        'X_test': preprocessing.landmarksToEmbedding(dataset['X_test']),
        'Y_test': dataset['Y_test'],
        'X_train': preprocessing.landmarksToEmbedding(dataset['X_train']),
        'Y_train': dataset['Y_train'],
    }


def main():
    fnames = [x[5:] for x in os.listdir('datasets/') if x.startswith('pose_')]
    print('[INFO]: found %d pose files' % len(fnames))

    for fname in fnames:
        fpathPose = os.path.join('datasets', 'pose_' + fname)

        poses = utils.load_pickle(fpathPose)
        dataset = postprocess(poses, testSplit=0.1)

        fpathDS = os.path.join('datasets', fname)
        print('[INFO]: creating dataset %s from poses %s' % (fpathDS, fpathPose))
        utils.save_pickle(dataset, fpathDS)

        if 'mediapipe' in fname:
            fpathDS = os.path.join('datasets', 'selected_' + fname)
            print('[INFO]: creating dataset (selected keypoints) %s from poses %s' % (fpathDS, fpathPose))
            dataset = selectKeypointsMediapipe(dataset)
            utils.save_pickle(dataset, fpathDS)

        if 'mediapipe' in fname or 'movenet' in fname:
            fpathDS = os.path.join('datasets', 'normalized_' + fname)
            print('[INFO]: creating dataset (normalized) %s from poses %s' % (fpathDS, fpathPose))
            dataset = normalizeDataset(dataset)
            utils.save_pickle(dataset, fpathDS)


if __name__ == '__main__':
    main()
