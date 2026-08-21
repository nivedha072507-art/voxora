"""Tests for the training script 02_train_model.py.

The script runs at import time, so each test writes a small synthetic dataset
into the sandbox directory and then loads the script.
"""

import csv
import pickle

import numpy as np
import pytest

DATASET = "voxera_7signs_data.csv"
MODEL = "voxera_7signs_model.pkl"
SIGNS = ["YES", "NO", "WHEN", "WHERE", "STOP", "HELP", "WHAT"]
SAMPLES_PER_SIGN = 6


@pytest.fixture
def dataset(sandbox):
    rng = np.random.default_rng(0)
    with open(sandbox / DATASET, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"coord_{i}" for i in range(126)] + ["label"])
        for index, sign in enumerate(SIGNS):
            centre = rng.normal(size=126) * 5
            for _ in range(SAMPLES_PER_SIGN):
                row = centre + rng.normal(scale=0.01, size=126)
                writer.writerow(list(row) + [sign])
    return sandbox


def test_training_writes_a_usable_model(dataset, load_script):
    load_script("train_model")

    with open(dataset / MODEL, "rb") as f:
        model = pickle.load(f)

    assert sorted(model.classes_) == sorted(SIGNS)
    assert model.n_features_in_ == 126


def test_trained_model_predicts_the_training_signs(dataset, load_script):
    load_script("train_model")

    with open(dataset / MODEL, "rb") as f:
        model = pickle.load(f)

    with open(dataset / DATASET, newline="") as f:
        frame = next(row for row in csv.reader(f) if row[-1] in SIGNS)
    features = np.array([float(value) for value in frame[:-1]])

    assert model.predict([features])[0] == frame[-1]


def test_training_reports_accuracy(dataset, load_script, capsys):
    load_script("train_model")

    output = capsys.readouterr().out
    assert "Model Accuracy:" in output
    assert MODEL in output


def test_training_uses_a_reproducible_split(dataset, load_script):
    module = load_script("train_model")

    assert len(module.X_test) == pytest.approx(
        0.2 * len(SIGNS) * SAMPLES_PER_SIGN, abs=1
    )
    assert "label" not in module.X.columns
