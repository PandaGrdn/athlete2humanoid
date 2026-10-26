"""Day 2: MediaPipe Pose on SAM-cropped frames -> smoothed (frames, 33, 3) world landmarks.

Use the MediaPipe *Tasks* API (PoseLandmarker), not the old mp.solutions.pose. Recent
mediapipe releases dropped the legacy solutions. Model file:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
"""

import numpy as np


def landmarks_from_crops(crops, fps):
    """Run PoseLandmarker in VIDEO mode; return (frames, 33, 3) pose_world_landmarks (metres,
    hip-centred) and (frames, 33) visibility. Fill missed frames with NaN."""
    raise NotImplementedError("Day 2")


def smooth(landmarks, window=7):
    """Day 2: interpolate NaNs, then Savitzky-Golay (scipy.signal.savgol_filter) along time."""
    raise NotImplementedError("Day 2")


def save(landmarks, path):
    np.save(path, landmarks)
