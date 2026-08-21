"""Import-time stubs for the hardware dependencies of the VOXORA scripts.

The scripts import ``cv2``, ``mediapipe``, ``speech_recognition`` and
``pyaudio`` at module scope and immediately touch a camera or a microphone.
These stubs make the modules importable in a headless test environment while
recording the calls the scripts make, so the pure logic around them can be
asserted on.
"""

import sys
import types
from typing import ClassVar

import numpy as np

FONT_HERSHEY_SIMPLEX = 0
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 5


class FakeVideoCapture:
    """Camera that yields a fixed number of frames and then stops."""

    def __init__(self, index=0, frames=0, frame=None, opened=True):
        self.index = index
        self.frames_remaining = frames
        self.frame = frame
        self._opened = opened
        self.released = False
        self.properties = {}

    def isOpened(self):
        return self._opened

    def read(self):
        if self.frames_remaining <= 0:
            return False, None
        self.frames_remaining -= 1
        frame = self.frame
        if frame is None:
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
        return True, frame.copy()

    def set(self, prop, value):
        self.properties[prop] = value
        return True

    def get(self, prop):
        return self.properties.get(prop, 0)

    def release(self):
        self.released = True
        self._opened = False


def _make_cv2():
    cv2 = types.ModuleType("cv2")

    cv2.FONT_HERSHEY_SIMPLEX = FONT_HERSHEY_SIMPLEX
    cv2.COLOR_BGR2RGB = COLOR_BGR2RGB
    cv2.COLOR_RGB2BGR = COLOR_RGB2BGR
    cv2.WINDOW_NORMAL = 0
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    cv2.CAP_PROP_FPS = 5

    cv2.calls = []

    def _record(name):
        def recorder(*args, **kwargs):
            cv2.calls.append((name, args, kwargs))

        return recorder

    def cvtColor(img, code):
        # Both conversions the scripts use are a channel reversal.
        return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])

    def flip(img, axis):
        cv2.calls.append(("flip", (axis,), {}))
        return np.asarray(img)[:, ::-1] if axis == 1 else np.asarray(img)[::-1]

    cv2.cvtColor = cvtColor
    cv2.flip = flip
    cv2.putText = _record("putText")
    cv2.rectangle = _record("rectangle")
    cv2.line = _record("line")
    cv2.imshow = _record("imshow")
    cv2.namedWindow = _record("namedWindow")
    cv2.resizeWindow = _record("resizeWindow")
    cv2.destroyAllWindows = _record("destroyAllWindows")
    # Tests override ``keys`` to script the keyboard input of the main loops.
    cv2.keys = [ord("q")]

    def waitKey(delay=0):
        if len(cv2.keys) > 1:
            return cv2.keys.pop(0)
        return cv2.keys[0]

    cv2.waitKey = waitKey
    cv2.VideoCapture = FakeVideoCapture
    return cv2


class FakeHands:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.results = None

    def process(self, image):
        return self.results

    def close(self):
        self.closed = True


def _make_mediapipe():
    mediapipe = types.ModuleType("mediapipe")
    hands_module = types.SimpleNamespace(
        Hands=FakeHands,
        HAND_CONNECTIONS=object(),
    )
    drawing_utils = types.SimpleNamespace(draw_landmarks=lambda *a, **k: None)
    mediapipe.solutions = types.SimpleNamespace(
        hands=hands_module,
        drawing_utils=drawing_utils,
    )
    return mediapipe


class UnknownValueError(Exception):
    pass


class RequestError(Exception):
    pass


class FakeRecognizer:
    """Recognizer whose ``recognize_google`` result is scripted by the test."""

    microphone_names: ClassVar[list] = [
        "Speakers (Realtek)",
        "Microphone Array (Realtek Audio)",
    ]

    # When set, calibration reports this ambient noise floor.
    ambient_energy = None

    def __init__(self):
        self.dynamic_energy_threshold = False
        self.energy_threshold = 0
        self.result = ""
        self.error = None
        self.languages = []
        self.ambient_calls = []
        self.background_calls = []
        self.stop_fn = lambda wait_for_stop=True: None

    def recognize_google(self, audio, language=None):
        self.languages.append(language)
        if self.error is not None:
            raise self.error
        return self.result

    def adjust_for_ambient_noise(self, source, duration=1.0):
        self.ambient_calls.append(duration)
        if type(self).ambient_energy is not None:
            self.energy_threshold = type(self).ambient_energy

    def listen_in_background(self, source, callback, **kwargs):
        self.background_calls.append((source, callback, kwargs))
        return self.stop_fn


class FakeMicrophone:
    instances: ClassVar[list] = []
    available_names: ClassVar[list] = FakeRecognizer.microphone_names
    raise_on_init = None

    def __init__(self, device_index=None):
        if FakeMicrophone.raise_on_init is not None:
            raise FakeMicrophone.raise_on_init
        self.device_index = device_index
        FakeMicrophone.instances.append(self)

    @classmethod
    def list_microphone_names(cls):
        return cls.available_names

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _make_speech_recognition():
    module = types.ModuleType("speech_recognition")
    module.Recognizer = FakeRecognizer
    module.Microphone = FakeMicrophone
    module.UnknownValueError = UnknownValueError
    module.RequestError = RequestError
    return module


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reads = []
        self.stopped = False
        self.closed = False

    def read(self, frames, exception_on_overflow=True):
        self.reads.append(frames)
        return b"\x00" * frames * 2

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakePyAudio:
    instances: ClassVar[list] = []

    def __init__(self):
        self.streams = []
        self.terminated = False
        FakePyAudio.instances.append(self)

    def get_device_info_by_index(self, index):
        return {
            "name": f"Fake Input Device {index}",
            "defaultSampleRate": 48000.0,
            "maxInputChannels": 2,
        }

    def open(self, **kwargs):
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream

    def terminate(self):
        self.terminated = True


def _make_pyaudio():
    module = types.ModuleType("pyaudio")
    module.paInt16 = 8
    module.PyAudio = FakePyAudio
    return module


def install():
    """Register the stub modules, replacing any previously installed stub."""
    modules = {
        "cv2": _make_cv2(),
        "mediapipe": _make_mediapipe(),
        "speech_recognition": _make_speech_recognition(),
        "pyaudio": _make_pyaudio(),
    }
    sys.modules.update(modules)
    FakeMicrophone.instances = []
    FakeMicrophone.raise_on_init = None
    FakeMicrophone.available_names = FakeRecognizer.microphone_names
    FakeRecognizer.ambient_energy = None
    FakePyAudio.instances = []
    return modules
