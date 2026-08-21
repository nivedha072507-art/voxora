# Testing VOXORA

The VOXORA scripts were previously only available inside `voxora (2).zip`, and the
project had no tests. The scripts are now checked in at the repository root and
covered by a pytest suite.

## Running the suite

```bash
pip install -r requirements-dev.txt
pytest                                  # run everything
pytest --cov=. --cov-report=term-missing
```

No camera, microphone or trained model is required: `tests/stubs.py` registers
in-memory replacements for `cv2`, `mediapipe`, `speech_recognition` and
`pyaudio`, and every test runs in a scratch directory containing a stand-in
model pickle (`tests/fakes.py`).

## How the scripts are loaded

The scripts are standalone programs, not importable modules: their names start
with digits and they open a camera, microphone or model file at module scope.
`tests/conftest.py` therefore exposes a `load_script` fixture that loads a
single script from its file path into a fresh module object for each test, so
module-level globals (`latest_speech_text`, `prediction_buffer`, ...) never leak
between tests.

## What is covered

| Script | Covered behaviour |
| --- | --- |
| `01_collect_data.py` | CSV header layout, per-sign sample targets, recording only after `s`, skipping hand-less frames, quitting with `q` |
| `02_train_model.py` | training on a synthetic dataset, model persistence, reproducible split |
| `03_live_inference.py` | landmark features, confidence threshold, 6-of-10 majority vote, hand-loss grace period, camera cleanup |
| `04_speech_to_text.py` | recognition result publishing, ambient calibration, error handling, microphone failure |
| `05_voxora_main.py` | Tamil/English language switching, speech callback states, microphone selection and energy-threshold clamping, Tamil text wrapping/truncation, the render loop and cleanup |
| `05_voxora_main_backup.py` | Tamil-only speech callback, local font fallback, render loop, cleanup |
| `mic_test.py` | 16 kHz mono stream configuration, 5 s capture, resource teardown |

The only lines left uncovered are the `if __name__ == "__main__":` entry points.
