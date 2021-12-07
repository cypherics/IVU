import os
import cv2
import numpy as np

import utils


def extract_mediapipe(datasetPath):
    """Mediapipe provides 3D pose estimation
    """

    import mediapipe as mp
    
    labels = list(os.listdir(datasetPath))
    dataset = dict()

    print('[INFO]: extracting poses from dataset using mediapipe...')
    
    with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
            
        print('[INFO]: found %d classes' % len(labels))

        for label in labels:
            labelPath = os.path.join(datasetPath, label)
            fnames = list(os.listdir(labelPath))
            dataset[label] = list()

            print('[INFO]: processing class %s...' % label)
            print('[INFO]: progress: 0 / %d' % len(fnames), end='\r')
            
            for i, fname in enumerate(fnames):
                fpath = os.path.join(labelPath, fname)
                image = cv2.imread(fpath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                results = pose.process(image)
                if not results.pose_landmarks:
                    print('[INFO]: progress: %d / %d' % (i, len(fnames)), end='\r')
                    continue

                # visibility might be useful
                # landmarks = [(p.x, p.y, p.z, p.visibility) for p in results.pose_landmarks]
                keypoints = [(p.x, p.y, p.z) for p in results.pose_landmarks.landmark]
                keypoints = np.array(keypoints)

                dataset[label].append(keypoints)
                print('[INFO]: progress: %d / %d' % (i, len(fnames)), end='\r')

            dataset[label] = np.array(dataset[label], dtype=np.float32)
            print('[INFO]: progress: %d / %d' % (len(fnames), len(fnames)))

    return dataset
    

def extract_ActionAI(datasetPath):
    """ActionAI provides 2D pose estimation
    """

    import tensorflow as tf
    
    labels = list(os.listdir(datasetPath))
    dataset = dict()

    print('[INFO]: extracting poses from dataset using ActionAI...')

    interpreter = tf.lite.Interpreter(model_path='./models/existing/ActionAIpose.tflite')
    interpreter.allocate_tensors()
    
    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()
    _, mpDim, _, numKeyPoints = outputDetails[0]['shape']
    imageInputSize = (inputDetails[0]['shape'][1], inputDetails[0]['shape'][2])

    print('[INFO]: found %d classes' % len(labels))
    
    for label in labels:
        labelPath = os.path.join(datasetPath, label)
        fnames = list(os.listdir(labelPath))
        dataset[label] = list()

        print('[INFO]: processing class %s...' % label)
        print('[INFO]: progress: 0 / %d' % len(fnames), end='\r')
        
        for i, fname in enumerate(fnames):
            fpath = os.path.join(labelPath, fname)
            image = cv2.imread(fpath)

            # resize image for model
            image = cv2.resize(image, imageInputSize, interpolation=cv2.INTER_NEAREST)
            image = np.expand_dims(image.astype(inputDetails[0]['dtype'])[:, :, :3], axis=0)
            
            # run model
            interpreter.set_tensor(inputDetails[0]['index'], image)
            interpreter.invoke()
            result = interpreter.get_tensor(outputDetails[0]['index'])
            
            res = result.reshape(1, mpDim**2, numKeyPoints)
            coords = np.divmod(np.argmax(res, axis=1), mpDim)
            
            keypoints = np.vstack(coords).T

            dataset[label].append(keypoints)
            print('[INFO]: progress: %d / %d' % (i, len(fnames)), end='\r')

        dataset[label] = np.array(dataset[label], dtype=np.float32)
        print('[INFO]: progress: %d / %d' % (len(fnames), len(fnames)))

    return dataset


def postprocess_dataset(dataset):
    X = list()
    Y = list()
    labels = list()

    for i, label in enumerate(dataset):
        keypoints = dataset[label]
        X.append(keypoints)
        Y.append(np.full(keypoints.shape[0], i, dtype=np.float32))
        labels.append(label)

    dataset = {
        'labels': labels,
        'X': np.concatenate(X, axis=0),
        'Y': np.concatenate(Y, axis=0),
    }

    return dataset


def main(datasetsPath):
    dataset1Path = os.path.join(datasetsPath, 'dataset1/')
    dataset2Path = os.path.join(datasetsPath, 'dataset2/train')

    datasetPaths = [dataset1Path, dataset2Path]
    
    poseExtractors = {
        'mediapipe': extract_mediapipe,
        'actionai': extract_ActionAI,
    }

    for i, datasetPath in enumerate(datasetPaths):

        for poseExtractorName in poseExtractors:
            outputPath = 'datasets/ds%d_%s' % (i+1, poseExtractorName)

            if not os.path.exists(outputPath):
                poseExtractorFunc = poseExtractors[poseExtractorName]
                
                dataset = poseExtractorFunc(datasetPath)
                dataset = postprocess_dataset(dataset)
                print('[INFO]: saving dataset to: %s' % outputPath)
                utils.save_pickle(dataset, outputPath)

    ###########################################################################
    # # Dataset2 - ActionAI
    # outputPath = 'datasets/ds2_actionai.pkl'
    # if not os.path.exists(outputPath):
    #     dataset = extract_ActionAI(dataset2Path)
    #     dataset = postprocess_dataset(dataset)
    #     print('[INFO]: saving dataset to: %s' % outputPath)
    #     utils.save_pickle(dataset, outputPath)

    # outputPath = 'datasets/ds2_mediapipe.pkl'
    # if not os.path.exists(outputPath):
    #     dataset = extract_mediapipe(datasetPath2)
    #     dataset = postprocess_dataset(dataset)
    #     fpath = 'datasets/ds2_mediapipe.pkl'
    #     print('[INFO]: saving dataset to: %s' % fpath)
    #     utils.save_pickle(dataset, fpath)


if __name__ == '__main__':
    main('F:\Studium\IVU\KU\project\datasets')
