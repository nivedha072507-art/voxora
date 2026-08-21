"""Tests for the dataset writer in 01_collect_data.py."""

import csv

import pytest
import stubs
from fakes import Results, make_hand


@pytest.fixture
def collector(load_script):
    return load_script("collect_data")


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_import_writes_a_header_of_landmark_columns_and_a_label(collector, sandbox):
    header = _rows(sandbox / collector.CSV_FILE)[0]

    assert header[-1] == "label"
    assert header[:3] == ["coord_0", "coord_1", "coord_2"]
    assert len(header) == 127


def test_seven_signs_are_collected(collector):
    assert collector.SIGNS == [
        "YES",
        "NO",
        "WHEN",
        "WHERE",
        "STOP",
        "HELP",
        "WHAT",
    ]


def test_nothing_is_recorded_until_the_start_key_is_pressed(collector, sandbox):
    collector.SIGNS = ["YES"]
    collector.SAMPLES_PER_SIGN = 3
    collector.hands.results = Results([make_hand(offset=0.2)])
    collector.cv2.keys = [ord("x")]
    collector.cv2.VideoCapture = lambda index: stubs.FakeVideoCapture(frames=5)

    collector.collect_data()

    assert len(_rows(sandbox / collector.CSV_FILE)) == 1


def test_samples_are_appended_once_recording_starts(collector, sandbox):
    collector.SIGNS = ["YES"]
    collector.SAMPLES_PER_SIGN = 2
    collector.hands.results = Results([make_hand(offset=0.2)])
    collector.cv2.keys = [ord("s")]
    collector.cv2.VideoCapture = lambda index: stubs.FakeVideoCapture(frames=10)

    collector.collect_data()

    rows = _rows(sandbox / collector.CSV_FILE)
    assert len(rows) == 3
    for row in rows[1:]:
        assert row[-1] == "YES"
        assert len(row) == 127


def test_recording_stops_at_the_sample_target_per_sign(collector, sandbox):
    collector.SIGNS = ["YES", "NO"]
    collector.SAMPLES_PER_SIGN = 2
    collector.hands.results = Results([make_hand(offset=0.2)])
    collector.cv2.keys = [ord("s")]
    collector.cv2.VideoCapture = lambda index: stubs.FakeVideoCapture(frames=50)

    collector.collect_data()

    labels = [row[-1] for row in _rows(sandbox / collector.CSV_FILE)[1:]]
    assert labels == ["YES", "YES", "NO", "NO"]


def test_frames_without_hands_are_skipped(collector, sandbox):
    collector.SIGNS = ["YES"]
    collector.SAMPLES_PER_SIGN = 2
    collector.hands.results = Results(None)
    collector.cv2.keys = [ord("s")]
    collector.cv2.VideoCapture = lambda index: stubs.FakeVideoCapture(frames=6)

    collector.collect_data()

    assert len(_rows(sandbox / collector.CSV_FILE)) == 1


def test_quit_key_ends_the_session_and_releases_the_camera(collector, sandbox):
    camera = stubs.FakeVideoCapture(frames=20)
    collector.SIGNS = ["YES"]
    collector.SAMPLES_PER_SIGN = 5
    collector.hands.results = Results([make_hand(offset=0.2)])
    collector.cv2.keys = [ord("q")]
    collector.cv2.VideoCapture = lambda index: camera

    collector.collect_data()

    assert camera.released
    assert len(_rows(sandbox / collector.CSV_FILE)) == 1
