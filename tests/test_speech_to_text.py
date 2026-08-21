"""Tests for 04_speech_to_text.py, the standalone speech recognition helper."""

import pytest

import stubs


@pytest.fixture
def stt(load_script):
    return load_script("speech_to_text")


def test_recognized_text_is_published(stt):
    recognizer = stt.sr.Recognizer()
    recognizer.result = "turn on the light"

    stt.speech_callback(recognizer, audio=object())

    assert stt.latest_speech_text == "turn on the light"


def test_unintelligible_audio_keeps_the_previous_text(stt):
    stt.latest_speech_text = "earlier phrase"
    recognizer = stt.sr.Recognizer()
    recognizer.error = stubs.UnknownValueError()

    stt.speech_callback(recognizer, audio=object())

    assert stt.latest_speech_text == "earlier phrase"


def test_service_error_keeps_the_previous_text(stt, capsys):
    stt.latest_speech_text = "earlier phrase"
    recognizer = stt.sr.Recognizer()
    recognizer.error = stubs.RequestError("quota exceeded")

    stt.speech_callback(recognizer, audio=object())

    assert stt.latest_speech_text == "earlier phrase"
    assert "quota exceeded" in capsys.readouterr().out


def test_start_returns_the_background_stop_handle(stt):
    stop = stt.start_speech_recognition()

    assert callable(stop)


def test_start_calibrates_for_ambient_noise_before_listening(stt, monkeypatch):
    captured = []
    monkeypatch.setattr(
        stubs.FakeRecognizer,
        "listen_in_background",
        lambda self, source, callback, **kwargs: captured.append((self, callback))
        or (lambda **k: None),
    )

    stt.start_speech_recognition()

    recognizer, callback = captured[-1]
    assert recognizer.ambient_calls == [2]
    assert recognizer.dynamic_energy_threshold is True
    assert callback is stt.speech_callback


def test_microphone_failure_returns_none(stt, monkeypatch):
    monkeypatch.setattr(
        stubs.FakeMicrophone, "raise_on_init", OSError("no input device")
    )

    assert stt.start_speech_recognition() is None
