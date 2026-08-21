"""Tests for the Tamil-only behaviour kept in 05_voxora_main_backup.py."""

import numpy as np
import pytest
import stubs
from fakes import FakeModel, Results, make_hand


@pytest.fixture
def backup(load_script):
    return load_script("voxora_main_backup")


def test_missing_model_file_aborts_startup(load_script, sandbox, capsys):
    (sandbox / "voxera_7signs_model.pkl").unlink()

    with pytest.raises(SystemExit):
        load_script("voxora_main_backup")

    assert "Could not load model file" in capsys.readouterr().out


def test_speech_is_always_recognized_as_tamil(backup):
    recognizer = backup.sr.Recognizer()
    recognizer.result = "வணக்கம்"

    backup.speech_callback(recognizer, audio=object())

    assert backup.latest_speech_text == "வணக்கம்"
    assert recognizer.languages == ["ta-IN"]


def test_unintelligible_audio_keeps_the_previous_text(backup):
    backup.latest_speech_text = "முந்தைய"
    recognizer = backup.sr.Recognizer()
    recognizer.error = stubs.UnknownValueError()

    backup.speech_callback(recognizer, audio=object())

    assert backup.latest_speech_text == "முந்தைய"


def test_service_error_is_surfaced_in_the_overlay(backup):
    recognizer = backup.sr.Recognizer()
    recognizer.error = stubs.RequestError("offline")

    backup.speech_callback(recognizer, audio=object())

    assert backup.latest_speech_text == "Speech Service Error"


def test_missing_camera_stops_the_listener_and_returns(backup, monkeypatch):
    stopped = []
    monkeypatch.setattr(
        backup, "start_speech_recognition", lambda: lambda **k: stopped.append(k)
    )
    monkeypatch.setattr(
        backup.cv2, "VideoCapture", lambda index: stubs.FakeVideoCapture(opened=False)
    )

    backup.main()

    assert stopped == [{"wait_for_stop": False}]


def test_listener_starts_after_ambient_calibration(backup, monkeypatch):
    captured = []
    monkeypatch.setattr(
        stubs.FakeRecognizer,
        "listen_in_background",
        lambda self, source, callback, **kwargs: captured.append((self, callback))
        or (lambda **k: None),
    )

    assert callable(backup.start_speech_recognition())

    recognizer, callback = captured[-1]
    assert recognizer.ambient_calls == [1.5]
    assert callback is backup.speech_callback


def test_microphone_failure_returns_none(backup, monkeypatch):
    monkeypatch.setattr(
        stubs.FakeMicrophone, "raise_on_init", OSError("no input device")
    )

    assert backup.start_speech_recognition() is None


def test_local_tamil_font_is_used_when_present(backup, sandbox, monkeypatch):
    (sandbox / "tamil_font.ttf").write_bytes(b"not a font")
    fallback = backup.ImageFont.load_default()
    loaded = []

    def truetype(path, size):
        loaded.append(path)
        return fallback

    monkeypatch.setattr(backup.ImageFont, "truetype", truetype)

    backup.draw_tamil_text(np.zeros((60, 120, 3), dtype=np.uint8), "test", (0, 0))

    assert loaded == ["tamil_font.ttf"]


def test_main_renders_the_sign_and_speech_panels(backup, monkeypatch):
    monkeypatch.setattr(backup, "start_speech_recognition", lambda: None)
    backup.model = FakeModel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    backup.hands.results = Results([make_hand(offset=0.2)])
    backup.latest_speech_text = "வணக்கம்"
    camera = stubs.FakeVideoCapture(frames=6)
    monkeypatch.setattr(backup.cv2, "VideoCapture", lambda index: camera)
    backup.cv2.keys = [ord("x")] * 6 + [ord("q")]

    backup.main()

    rendered = [args[1] for name, args, _ in backup.cv2.calls if name == "putText"]
    assert "Sign: YES" in rendered
    assert camera.released


def test_rendering_returns_a_frame_of_the_same_shape(backup):
    frame = np.zeros((120, 320, 3), dtype=np.uint8)

    result = backup.draw_tamil_text(frame, "Speech: வணக்கம்", (10, 10))

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
