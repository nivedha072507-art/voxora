"""Picklable fakes used to stand in for the trained ExtraTrees model."""

import numpy as np


class FakeModel:
    """Minimal predict_proba/classes_ surface used by the inference scripts."""

    classes_ = np.array(["HELP", "NO", "STOP", "WHAT", "WHEN", "WHERE", "YES"])

    def __init__(self, probabilities=None):
        self.probabilities = probabilities
        self.calls = []

    def predict_proba(self, features):
        self.calls.append(np.asarray(features))
        if self.probabilities is not None:
            return np.array([self.probabilities])
        return np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])


class Landmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class Hand:
    """Stand-in for a MediaPipe `NormalizedLandmarkList` (21 landmarks)."""

    def __init__(self, landmarks):
        self.landmark = landmarks


class Results:
    def __init__(self, multi_hand_landmarks=None):
        self.multi_hand_landmarks = multi_hand_landmarks


def make_hand(offset=0.0, scale=0.1, z_offset=0.0):
    """Build a hand whose landmarks are spaced so the wrist->MCP distance is known."""
    landmarks = []
    for i in range(21):
        landmarks.append(
            Landmark(
                x=offset + i * scale,
                y=offset + i * scale,
                z=z_offset,
            )
        )
    return Hand(landmarks)


def make_degenerate_hand(x=0.5):
    """All landmarks identical, which makes the normalization scale zero."""
    return Hand([Landmark(x, 0.5, 0.0) for _ in range(21)])
