import enum
from enum import Enum
import numpy as np


class MediaPipePoseLandmarks(Enum):
    Nose = 0
    LeftEyeInner = 1
    LeftEye = 2
    LeftEyeOuter = 3
    RightEyeInner = 4
    RightEye = 5
    RightEyeOuter = 6
    LeftEar = 7
    RightEar = 8
    MouthLeft = 9
    MouthRight = 10
    LeftShoulder = 11
    RightShoulder = 12
    LeftElbow = 13
    RightElbow = 14
    LeftWrist = 15
    RightWrist = 16
    LeftPinky = 17
    RightPinky = 18
    LeftIndex = 19
    RightIndex = 20
    LeftThumb = 21
    RightThumb = 22
    LeftHip = 23
    RightHip = 24
    LeftKnee = 25
    RightKnee = 26
    LeftAnkle = 27
    RightAnkle = 28
    LeftHeel = 29
    RightHeel = 30
    LeftFootIndex = 31
    RightFootIndex = 32


class Pose16LandmarksBodyModel(Enum):
    Nose = 0
    LeftEye = 1
    RightEye = 2
    LeftEar = 3
    RightEar = 4
    LeftShoulder = 5
    RightShoulder = 6
    LeftElbow = 7
    RightElbow = 8
    LeftWrist = 9
    RightWrist = 10
    LeftHip = 11
    RightHip = 12
    LeftKnee = 13
    RightKnee = 14
    LeftAnkle = 15
    RightAnkle = 16


def get_pose_size(
    landmarks: np.ndarray, body_model, torso_size_multiplier: float = 2.5
):
    assert type(body_model) == enum.EnumMeta

    # landmarks.shape: [N_poses, N_landmarks, N_dim]
    n_dim = landmarks.shape[-1]

    hips_center = (
        landmarks[:, body_model.LeftHip.value] + landmarks[:, body_model.RightHip.value]
    ) / 2
    shoulders_center = (
        landmarks[:, body_model.LeftShoulder.value]
        + landmarks[:, body_model.RightShoulder.value]
    ) / 2

    torso_size = np.linalg.norm(shoulders_center - hips_center, axis=-1)

    pose_center = hips_center
    dist = np.linalg.norm(landmarks - pose_center.reshape(-1, 1, n_dim), axis=-1)
    max_dist = np.max(dist, axis=-1)

    pose_size = np.maximum(torso_size * torso_size_multiplier, max_dist)
    # pose_size.shape: [N_poses]

    return pose_size


def normalize_landmarks(landmarks: np.ndarray, body_model) -> np.ndarray:
    assert type(body_model) == enum.EnumMeta

    # landmarks.shape: [N_poses, N_landmarks, N_dim]
    n_dim = landmarks.shape[-1]
    pose_center = (
        landmarks[:, body_model.LeftHip.value] + landmarks[:, body_model.RightHip.value]
    ) / 2
    landmarks -= pose_center.reshape(-1, 1, n_dim)
    landmarks /= get_pose_size(landmarks, body_model=body_model).reshape(-1, 1, 1)

    return landmarks


def landmarks_to_embedding(landmarks: np.ndarray, body_model) -> np.ndarray:
    assert type(body_model) == enum.EnumMeta
    landmarks = normalize_landmarks(
        np.expand_dims(landmarks, axis=0) if landmarks.ndim == 2 else landmarks,
        body_model,
    )
    return landmarks
