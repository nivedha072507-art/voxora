"""Loading, saving, and querying the sign classification model."""

import pickle

import numpy as np

from .config import MODEL_PATH


def load_model(path=MODEL_PATH, exit_on_error=False):
    """Load the pickled classifier.

    With ``exit_on_error`` the process exits after reporting a load failure,
    which is what the interactive scripts want.
    """
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[INFO] Loaded model: {path}")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load model file '{path}': {e}")
        if exit_on_error:
            raise SystemExit(1)
        raise


def save_model(model, path=MODEL_PATH):
    """Persist the classifier as a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved '{path}'")


def predict_sign(model, features):
    """Return the most likely sign label and its probability."""
    probs = model.predict_proba([features])[0]
    return model.classes_[np.argmax(probs)], float(np.max(probs))
