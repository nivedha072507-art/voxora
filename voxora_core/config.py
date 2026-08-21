"""Shared configuration for the VOXORA scripts."""

SIGNS = ["YES", "NO", "WHEN", "WHERE", "STOP", "HELP", "WHAT"]

SIGN_TAMIL = {
    "YES": "ஆம்",
    "NO": "இல்லை",
    "WHAT": "என்ன",
    "WHERE": "எங்கே",
    "HELP": "உதவி",
    "WHEN": "எப்போது",
    "STOP": "நிறுத்து",
}

DATA_CSV_PATH = "voxera_7signs_data.csv"
MODEL_PATH = "voxera_7signs_model.pkl"

LANDMARKS_PER_HAND = 21
COORDS_PER_HAND = LANDMARKS_PER_HAND * 3
MAX_HANDS = 2
FEATURE_COUNT = COORDS_PER_HAND * MAX_HANDS

CONFIDENCE_THRESHOLD = 0.60
PREDICTION_BUFFER_SIZE = 10
PREDICTION_VOTES_REQUIRED = 6
MISSING_HAND_GRACE_FRAMES = 12
