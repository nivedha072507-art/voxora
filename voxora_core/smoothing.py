"""Temporal smoothing of frame-by-frame sign predictions."""

from collections import Counter, deque

from .config import (
    CONFIDENCE_THRESHOLD,
    MISSING_HAND_GRACE_FRAMES,
    PREDICTION_BUFFER_SIZE,
    PREDICTION_VOTES_REQUIRED,
)


class SignSmoother:
    """Majority-vote a sliding window of predictions into a stable label.

    A prediction only enters the window when the model is confident enough, and
    the label only changes once enough frames in the window agree. The
    "no hand" state is delayed by a grace period so brief tracking losses do
    not make the displayed label flicker.
    """

    def __init__(
        self,
        buffer_size=PREDICTION_BUFFER_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        votes_required=PREDICTION_VOTES_REQUIRED,
        grace_frames=MISSING_HAND_GRACE_FRAMES,
        initial_label="Waiting...",
        missing_label="No Hand Detected",
    ):
        self.buffer = deque(maxlen=buffer_size)
        self.confidence_threshold = confidence_threshold
        self.votes_required = votes_required
        self.grace_frames = grace_frames
        self.missing_label = missing_label
        self.label = initial_label
        self.missing_frames = 0

    def update(self, prediction, confidence):
        """Feed one detected-hand prediction and return the label to display."""
        self.missing_frames = 0

        if confidence > self.confidence_threshold:
            self.buffer.append(prediction)

        if self.buffer:
            most_common, count = Counter(self.buffer).most_common(1)[0]
            if count >= self.votes_required:
                self.label = most_common

        return self.label

    def update_missing(self):
        """Feed a frame without hands and return the label to display."""
        self.missing_frames += 1

        if self.missing_frames > self.grace_frames:
            self.buffer.clear()
            self.label = self.missing_label

        return self.label
