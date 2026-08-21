"""Tests for the prediction smoothing loop in 03_live_inference.py."""

import pytest

import stubs
from fakes import FakeModel, Results, make_hand

CONFIDENT_YES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
UNCERTAIN = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]


@pytest.fixture
def inference(load_script):
    return load_script("live_inference")


def _run(module, frames, results, probabilities=CONFIDENT_YES):
    module.model = FakeModel(probabilities)
    module.hands.results = results
    module.cv2.keys = [ord("x")] * frames + [ord("q")]
    camera = stubs.FakeVideoCapture(frames=frames)
    module.cv2.VideoCapture = lambda index: camera
    module.prediction_buffer.clear()
    module.missing_hand_counter = 0
    module.main()
    return camera


def _rendered(module):
    return [
        args[1]
        for name, args, _ in module.cv2.calls
        if name == "putText"
    ]


def test_missing_model_file_raises_on_import(load_script, sandbox):
    (sandbox / "voxera_7signs_model.pkl").unlink()

    with pytest.raises(FileNotFoundError):
        load_script("live_inference")


def test_model_is_loaded_from_the_pickle_next_to_the_script(inference):
    assert hasattr(inference.model, "predict_proba")
    assert inference.MODEL_PATH == "voxera_7signs_model.pkl"


def test_prediction_buffer_smooths_over_ten_frames(inference):
    assert inference.prediction_buffer.maxlen == 10


def test_sign_is_shown_once_six_of_ten_frames_agree(inference):
    hands = Results([make_hand(offset=0.2)])

    _run(inference, frames=6, results=hands)

    assert "Voxera Sign: YES" in _rendered(inference)


def test_sign_is_withheld_until_the_majority_is_reached(inference):
    hands = Results([make_hand(offset=0.2)])

    _run(inference, frames=5, results=hands)

    rendered = _rendered(inference)
    assert "Voxera Sign: Waiting..." in rendered
    assert "Voxera Sign: YES" not in rendered


def test_low_confidence_predictions_are_ignored(inference):
    hands = Results([make_hand(offset=0.2)])

    _run(inference, frames=10, results=hands, probabilities=UNCERTAIN)

    assert _rendered(inference) == ["Voxera Sign: Waiting..."] * 10


def test_brief_hand_loss_does_not_reset_the_display(inference):
    _run(inference, frames=12, results=Results(None))

    assert "Voxera Sign: No Hand Detected" not in _rendered(inference)


def test_prolonged_hand_loss_clears_the_display(inference):
    _run(inference, frames=13, results=Results(None))

    assert _rendered(inference)[-1] == "Voxera Sign: No Hand Detected"


def test_features_passed_to_the_model_match_the_extractor(inference):
    hands = Results([make_hand(offset=0.2)])

    _run(inference, frames=1, results=hands)

    features = inference.model.calls[0][0]
    assert len(features) == 126
    assert features == pytest.approx(inference.extract_landmarks_sorted(hands))


def test_camera_is_released_when_the_loop_ends(inference):
    camera = _run(inference, frames=2, results=Results(None))

    assert camera.released
