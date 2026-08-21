import importlib.util
import os
import pickle
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import fakes  # noqa: E402  (needs the sys.path entry above)
import stubs  # noqa: E402

SCRIPTS = {
    "collect_data": "01_collect_data.py",
    "train_model": "02_train_model.py",
    "mic_test": "mic_test.py",
    "live_inference": "03_live_inference.py",
    "speech_to_text": "04_speech_to_text.py",
    "voxora_main": "05_voxora_main.py",
    "voxora_main_backup": "05_voxora_main_backup.py",
}

MODEL_FILENAME = "voxera_7signs_model.pkl"


@pytest.fixture
def stub_modules():
    """Install fresh hardware stubs for a single test."""
    installed = stubs.install()
    yield installed
    for name in installed:
        sys.modules.pop(name, None)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Run the script inside a scratch directory holding a stand-in model file."""
    monkeypatch.chdir(tmp_path)
    with open(tmp_path / MODEL_FILENAME, "wb") as f:
        pickle.dump(fakes.FakeModel(), f)
    return tmp_path


@pytest.fixture
def load_script(stub_modules, sandbox):
    """Import one of the numbered VOXORA scripts as a fresh module object.

    The scripts are not importable packages (their names start with digits and
    they execute setup code at module scope), so each test gets its own module
    instance loaded straight from the file.
    """

    def _load(key):
        path = os.path.join(PROJECT_ROOT, SCRIPTS[key])
        spec = importlib.util.spec_from_file_location(f"voxora_{key}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        return module

    yield _load

    for key in SCRIPTS:
        sys.modules.pop(f"voxora_{key}", None)
