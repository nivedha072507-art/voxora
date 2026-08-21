"""Integrity-checked loading of the VOXORA sign classifier.

A pickle file is executable content: unpickling a file an attacker can write
runs arbitrary code. The model is therefore only unpickled after its SHA-256
digest matches the digest recorded in ``model_checksums.txt``.
"""

import hashlib
import os
import pickle
from pathlib import Path

DEFAULT_MODEL_NAME = "voxera_7signs_model.pkl"
CHECKSUM_FILE = "model_checksums.txt"

BASE_DIR = Path(__file__).resolve().parent


class ModelIntegrityError(Exception):
    """Raised when the model file is missing, unlisted or has an unexpected digest."""


def _read_expected_digests(checksum_path: Path) -> dict:
    digests = {}

    with checksum_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            digest, _, name = line.partition(" ")
            name = name.strip().lstrip("*")

            if digest and name:
                digests[name] = digest.lower()

    return digests


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_model_path(model_name: str = None) -> Path:
    """Resolve the model path, refusing anything outside the project directory."""

    model_name = model_name or os.environ.get("VOXORA_MODEL", DEFAULT_MODEL_NAME)
    path = (BASE_DIR / model_name).resolve()

    if BASE_DIR not in path.parents and path.parent != BASE_DIR:
        raise ModelIntegrityError(
            f"Model path '{model_name}' resolves outside the project directory."
        )

    return path


def load_model(model_name: str = None):
    """Unpickle the classifier only if its digest is the recorded one."""

    path = resolve_model_path(model_name)

    if not path.is_file():
        raise ModelIntegrityError(f"Model file not found: {path}")

    checksum_path = BASE_DIR / CHECKSUM_FILE

    if not checksum_path.is_file():
        raise ModelIntegrityError(f"Checksum file not found: {checksum_path}")

    expected = _read_expected_digests(checksum_path).get(path.name)

    if expected is None:
        raise ModelIntegrityError(
            f"No recorded SHA-256 for '{path.name}' in {CHECKSUM_FILE}. "
            "Add the digest of a model you trust before loading it."
        )

    actual = _sha256(path)

    if actual != expected:
        raise ModelIntegrityError(
            f"SHA-256 mismatch for '{path.name}'.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "Refusing to unpickle a model that is not the trusted one."
        )

    with path.open("rb") as handle:
        return pickle.load(handle)
