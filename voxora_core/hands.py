"""MediaPipe hand detection and landmark feature extraction."""

import mediapipe as mp
import numpy as np

from .config import COORDS_PER_HAND, MAX_HANDS

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def create_hand_detector(detection_confidence=0.5, tracking_confidence=None):
    """Create a MediaPipe Hands detector configured for VOXORA."""
    if tracking_confidence is None:
        tracking_confidence = detection_confidence

    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )


def extract_landmarks_sorted(results):
    """Return wrist-centred, scale-normalised landmarks for up to two hands.

    Hands are ordered by their wrist X position (left-to-right in the image) so
    that the feature vector layout is stable between collection and inference.
    Missing hands are zero-padded.
    """
    hand_features = [np.zeros(COORDS_PER_HAND) for _ in range(MAX_HANDS)]

    if results.multi_hand_landmarks:
        sorted_hands = sorted(
            results.multi_hand_landmarks,
            key=lambda lm: lm.landmark[0].x,
        )

        for idx, hand_landmarks in enumerate(sorted_hands[:MAX_HANDS]):
            base = hand_landmarks.landmark[0]
            mcp = hand_landmarks.landmark[9]

            scale = np.sqrt(
                (mcp.x - base.x) ** 2
                + (mcp.y - base.y) ** 2
                + (mcp.z - base.z) ** 2
            )
            if scale == 0:
                scale = 1.0

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([
                    (lm.x - base.x) / scale,
                    (lm.y - base.y) / scale,
                    (lm.z - base.z) / scale,
                ])

            hand_features[idx] = np.array(coords)

    return np.concatenate(hand_features)


def draw_hand_landmarks(frame, results):
    """Draw the detected hand skeletons onto a BGR frame in place."""
    if not results.multi_hand_landmarks:
        return frame

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
        )
    return frame
