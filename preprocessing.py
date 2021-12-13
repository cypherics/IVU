import numpy as np

from pose_estimation.pose_landmarks import PoseLandmarks


def getPoseSize(landmarks, torsoSizeMultiplier=2.5):
    # landmarks.shape: [N_poses, N_landmarks, N_dim]
    ndim = landmarks.shape[-1]

    hipsCenter = (landmarks[:, PoseLandmarks.LeftHip.value] + landmarks[:, PoseLandmarks.RightHip.value]) / 2
    shouldersCenter = (landmarks[:, PoseLandmarks.LeftShoulder.value] + landmarks[:, PoseLandmarks.RightShoulder.value]) / 2

    torsoSize = np.linalg.norm(shouldersCenter - hipsCenter, axis=-1)

    poseCenter = hipsCenter
    dist = np.linalg.norm(landmarks - poseCenter.reshape(-1, 1, ndim), axis=-1)
    maxDist = np.max(dist, axis=-1)

    poseSize = np.maximum(torsoSize * torsoSizeMultiplier, maxDist)
    # poseSize.shape: [N_poses]

    return poseSize


def normalizePoseLandmarks(landmarks):
    # landmarks.shape: [N_poses, N_landmarks, N_dim]
    ndim = landmarks.shape[-1]

    poseCenter = (landmarks[:, PoseLandmarks.LeftHip.value] + landmarks[:, PoseLandmarks.RightHip.value]) / 2

    landmarks -= poseCenter.reshape(-1, 1, ndim)
    landmarks /= getPoseSize(landmarks).reshape(-1, 1, 1)

    return landmarks


def landmarksToEmbedding(landmarks):
    landmarks = normalizePoseLandmarks(landmarks)
    landmarks = landmarks.reshape(landmarks.shape[0], -1)
    return landmarks
