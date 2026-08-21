"""Shared utilities for the VOXORA sign-language and speech scripts."""

from .camera import (
    camera_resolution,
    open_camera,
    read_mirrored_frame,
    release_camera,
)
from .config import (
    DATA_CSV_PATH,
    FEATURE_COUNT,
    MODEL_PATH,
    SIGN_TAMIL,
    SIGNS,
)
from .hands import (
    create_hand_detector,
    draw_hand_landmarks,
    extract_landmarks_sorted,
)
from .model import load_model, predict_sign, save_model
from .smoothing import SignSmoother
from .speech import SpeechListener
from .text import draw_tamil_text

__all__ = [
    "DATA_CSV_PATH",
    "FEATURE_COUNT",
    "MODEL_PATH",
    "SIGNS",
    "SIGN_TAMIL",
    "SignSmoother",
    "SpeechListener",
    "camera_resolution",
    "create_hand_detector",
    "draw_hand_landmarks",
    "draw_tamil_text",
    "extract_landmarks_sorted",
    "load_model",
    "open_camera",
    "predict_sign",
    "read_mirrored_frame",
    "release_camera",
    "save_model",
]
