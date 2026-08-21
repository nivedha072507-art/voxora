"""Tests for the language, speech and text-rendering logic of 05_voxora_main.py."""

import numpy as np
import pytest
import stubs
from fakes import FakeModel, Results, make_hand
from PIL import ImageDraw


@pytest.fixture
def main(load_script):
    return load_script("voxora_main")


def _recognizer(main):
    return main.sr.Recognizer()


def _blank_frame(width=800, height=600):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_missing_model_file_aborts_startup(load_script, sandbox, capsys):
    (sandbox / "voxera_7signs_model.pkl").unlink()

    with pytest.raises(SystemExit):
        load_script("voxora_main")

    assert "Could not load model file" in capsys.readouterr().out


# ------------------------------------------------------------------
# Sign vocabulary
# ------------------------------------------------------------------


def test_every_trained_sign_has_a_tamil_translation(main, load_script):
    trained_signs = load_script("collect_data").SIGNS

    assert sorted(main.SIGN_TAMIL) == sorted(trained_signs)


def test_unknown_sign_falls_back_to_the_raw_label(main):
    assert main.SIGN_TAMIL.get("Waiting...", "Waiting...") == "Waiting..."


# ------------------------------------------------------------------
# set_language
# ------------------------------------------------------------------


def test_a_selects_tamil(main):
    main.current_language = "en-IN"

    main.set_language(ord("a"))

    assert main.current_language == "ta-IN"
    assert main.language_name == "TAMIL"
    assert main.speech_status == "Ready - Tamil"


def test_s_selects_english(main):
    main.set_language(ord("s"))

    assert main.current_language == "en-IN"
    assert main.language_name == "ENGLISH"
    assert main.latest_speech_text == "Speak..."
    assert main.speech_status == "Ready - English"


def test_unrelated_key_leaves_the_language_untouched(main):
    main.set_language(ord("s"))

    main.set_language(ord("z"))

    assert main.current_language == "en-IN"
    assert main.language_name == "ENGLISH"


# ------------------------------------------------------------------
# speech_callback
# ------------------------------------------------------------------


def test_recognized_speech_is_stored_with_the_selected_language(main):
    main.set_language(ord("s"))
    recognizer = _recognizer(main)
    recognizer.result = "hello there"

    main.speech_callback(recognizer, audio=object())

    assert main.latest_speech_text == "hello there"
    assert main.speech_status == "Listening"
    assert recognizer.languages == ["en-IN"]


def test_blank_recognition_keeps_the_previous_text(main):
    main.latest_speech_text = "previous"
    recognizer = _recognizer(main)
    recognizer.result = "   "

    main.speech_callback(recognizer, audio=object())

    assert main.latest_speech_text == "previous"


def test_unintelligible_audio_reports_status_without_losing_text(main):
    main.latest_speech_text = "previous"
    recognizer = _recognizer(main)
    recognizer.error = stubs.UnknownValueError()

    main.speech_callback(recognizer, audio=object())

    assert main.latest_speech_text == "previous"
    assert main.speech_status == "Could not understand"


def test_service_failure_reports_status_without_losing_text(main):
    main.latest_speech_text = "previous"
    recognizer = _recognizer(main)
    recognizer.error = stubs.RequestError("no network")

    main.speech_callback(recognizer, audio=object())

    assert main.latest_speech_text == "previous"
    assert main.speech_status == "Speech service unavailable"


def test_unexpected_error_is_swallowed_and_marked_as_retrying(main):
    recognizer = _recognizer(main)
    recognizer.error = RuntimeError("boom")

    main.speech_callback(recognizer, audio=object())

    assert main.speech_status == "Speech retrying"


# ------------------------------------------------------------------
# start_speech_recognition
# ------------------------------------------------------------------


def test_microphone_array_is_preferred_over_the_default_device(main):
    stubs.FakeMicrophone.available_names = [
        "Speakers (Realtek)",
        "Stereo Mix",
        "Microphone Array (Realtek Audio)",
    ]

    stop = main.start_speech_recognition()

    assert callable(stop)
    assert stubs.FakeMicrophone.instances[-1].device_index == 2


def test_default_device_is_used_when_no_array_is_present(main):
    stubs.FakeMicrophone.available_names = ["Speakers (Realtek)", "Stereo Mix"]

    main.start_speech_recognition()

    assert stubs.FakeMicrophone.instances[-1].device_index is None


def test_calibration_clamps_the_energy_threshold_into_range(main, monkeypatch):
    captured = []
    monkeypatch.setattr(
        stubs.FakeRecognizer,
        "listen_in_background",
        lambda self, source, callback, **kwargs: captured.append(self) or (lambda **k: None),
    )

    monkeypatch.setattr(stubs.FakeRecognizer, "ambient_energy", 4000)
    main.start_speech_recognition()
    assert captured[-1].energy_threshold == 500

    monkeypatch.setattr(stubs.FakeRecognizer, "ambient_energy", 10)
    main.start_speech_recognition()
    assert captured[-1].energy_threshold == 120


def test_microphone_failure_returns_none(main, monkeypatch):
    monkeypatch.setattr(
        stubs.FakeMicrophone, "raise_on_init", OSError("no input device")
    )

    assert main.start_speech_recognition() is None


def test_unlistable_devices_still_falls_back_to_the_default_microphone(main, monkeypatch):
    def boom(cls):
        raise OSError("enumeration failed")

    monkeypatch.setattr(
        stubs.FakeMicrophone, "list_microphone_names", classmethod(boom)
    )

    assert callable(main.start_speech_recognition())
    assert stubs.FakeMicrophone.instances[-1].device_index is None


# ------------------------------------------------------------------
# draw_tamil_text
# ------------------------------------------------------------------


@pytest.fixture
def drawn_lines(main, monkeypatch):
    """Capture the lines handed to Pillow instead of inspecting pixels."""
    lines = []
    original = ImageDraw.Draw

    class Recorder:
        def __init__(self, image):
            self._draw = original(image)

        def textbbox(self, *args, **kwargs):
            return self._draw.textbbox(*args, **kwargs)

        def text(self, position, text, **kwargs):
            lines.append((position, text))
            return self._draw.text(position, text, **kwargs)

    monkeypatch.setattr(main.ImageDraw, "Draw", Recorder)
    return lines


def test_rendering_preserves_the_frame_shape_and_dtype(main):
    frame = _blank_frame()

    result = main.draw_tamil_text(frame, "VOXORA", (10, 10))

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_short_text_is_drawn_as_a_single_line(main, drawn_lines):
    main.draw_tamil_text(_blank_frame(), "hello world", (10, 20))

    assert drawn_lines == [((10, 20), "hello world")]


def test_long_text_wraps_within_the_max_width(main, drawn_lines):
    text = " ".join(["word"] * 40)

    main.draw_tamil_text(_blank_frame(), text, (0, 0), max_width=40, max_lines=5)

    assert len(drawn_lines) > 1
    assert " ".join(line for _, line in drawn_lines).startswith("word word")


def test_overflowing_text_is_truncated_with_an_ellipsis(main, drawn_lines):
    text = " ".join(["word"] * 40)

    main.draw_tamil_text(_blank_frame(), text, (0, 0), max_width=40, max_lines=2)

    assert len(drawn_lines) == 2
    assert drawn_lines[-1][1].endswith("...")


def test_lines_are_stacked_by_the_font_size_plus_spacing(main, drawn_lines):
    text = " ".join(["word"] * 8)

    main.draw_tamil_text(
        _blank_frame(), text, (5, 7), font_size=20, max_width=40, max_lines=3
    )

    ys = [position[1] for position, _ in drawn_lines]
    assert ys == [7, 7 + 34, 7 + 68]


def test_empty_text_draws_nothing(main, drawn_lines):
    main.draw_tamil_text(_blank_frame(), "   ", (0, 0))

    assert drawn_lines == []


def test_bold_variant_renders_without_the_windows_fonts(main, drawn_lines):
    main.draw_tamil_text(_blank_frame(), "ஆம்", (0, 0), bold=True)

    assert [text for _, text in drawn_lines] == ["ஆம்"]


# ------------------------------------------------------------------
# main loop
# ------------------------------------------------------------------

CONFIDENT_YES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
UNCERTAIN = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]


def _run_main(main, monkeypatch, frames, results, keys=None, probabilities=CONFIDENT_YES):
    stopped = []
    monkeypatch.setattr(
        main, "start_speech_recognition", lambda: lambda **k: stopped.append(k)
    )
    main.model = FakeModel(probabilities)
    main.hands.results = results
    main.cv2.keys = (keys or [ord("x")] * frames) + [ord("q")]
    camera = stubs.FakeVideoCapture(frames=frames)
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda index: camera)

    main.main()

    return camera, stopped


def _rendered(main):
    return [args[1] for name, args, _ in main.cv2.calls if name == "putText"]


def test_main_requests_a_720p_camera_feed(main, monkeypatch):
    camera, _ = _run_main(main, monkeypatch, frames=1, results=Results(None))

    assert camera.properties[main.cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert camera.properties[main.cv2.CAP_PROP_FRAME_HEIGHT] == 720


def test_main_stops_listening_and_returns_when_the_camera_is_missing(main, monkeypatch):
    stopped = []
    monkeypatch.setattr(
        main, "start_speech_recognition", lambda: lambda **k: stopped.append(k)
    )
    monkeypatch.setattr(
        main.cv2, "VideoCapture", lambda index: stubs.FakeVideoCapture(opened=False)
    )

    main.main()

    assert stopped == [{"wait_for_stop": False}]


def test_main_displays_a_sign_once_six_of_ten_frames_agree(main, monkeypatch):
    _run_main(main, monkeypatch, frames=6, results=Results([make_hand(offset=0.2)]))

    assert "Sign: YES" in _rendered(main)


def test_main_ignores_low_confidence_predictions(main, monkeypatch):
    _run_main(
        main,
        monkeypatch,
        frames=10,
        results=Results([make_hand(offset=0.2)]),
        probabilities=UNCERTAIN,
    )

    assert "Sign: YES" not in _rendered(main)
    assert "Sign: Waiting..." in _rendered(main)


def test_main_clears_the_sign_after_prolonged_hand_loss(main, monkeypatch):
    _run_main(main, monkeypatch, frames=13, results=Results(None))

    assert "Sign: No Hand Detected" in _rendered(main)


def test_main_keeps_the_sign_through_brief_hand_loss(main, monkeypatch):
    _run_main(main, monkeypatch, frames=12, results=Results(None))

    assert "Sign: No Hand Detected" not in _rendered(main)


def test_main_switches_language_from_the_keyboard(main, monkeypatch):
    _run_main(
        main,
        monkeypatch,
        frames=3,
        results=Results(None),
        keys=[ord("s"), ord("a"), ord("s")],
    )

    assert main.language_name == "ENGLISH"
    assert "Language: ENGLISH" in _rendered(main)


def test_main_releases_the_camera_and_microphone_on_exit(main, monkeypatch):
    camera, stopped = _run_main(main, monkeypatch, frames=2, results=Results(None))

    assert camera.released
    assert stopped == [{"wait_for_stop": False}]
    assert main.hands.closed
