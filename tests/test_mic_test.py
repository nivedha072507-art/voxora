"""Tests for the microphone smoke-check script mic_test.py."""

import math

import pytest
import stubs


@pytest.fixture
def mic(load_script):
    return load_script("mic_test")


def test_stream_is_opened_as_16khz_mono_input(mic):
    stream = stubs.FakePyAudio.instances[-1].streams[-1]

    assert stream.kwargs["rate"] == 16000
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["input"] is True
    assert stream.kwargs["input_device_index"] == mic.DEVICE


def test_five_seconds_of_audio_are_captured(mic):
    stream = stubs.FakePyAudio.instances[-1].streams[-1]

    assert len(stream.reads) == math.floor(16000 / 1024 * 5)
    assert set(stream.reads) == {1024}


def test_device_details_are_reported(load_script, capsys):
    load_script("mic_test")

    output = capsys.readouterr().out

    assert "Device: Fake Input Device 8" in output
    assert "Microphone test PASSED." in output


def test_resources_are_closed_after_the_check(mic):
    audio = stubs.FakePyAudio.instances[-1]

    assert audio.streams[-1].stopped
    assert audio.streams[-1].closed
    assert audio.terminated
