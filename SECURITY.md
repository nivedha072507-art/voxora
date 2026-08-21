# Security notes

VOXORA is a local desktop application (webcam + microphone). It exposes no HTTP
server, no database and no user accounts, so there is no attack surface for SQL
injection, CORS misconfiguration, debug endpoints or authentication bypass.
The relevant risks are supply-chain and data-handling ones.

## Model files are executable content

`voxera_7signs_model.pkl` is a Python pickle. Unpickling a file an attacker can
replace results in arbitrary code execution as the user running VOXORA. The
inference scripts therefore load the model through `model_loader.load_model()`,
which:

- resolves the path inside the project directory only (no traversal via
  `VOXORA_MODEL`), and
- verifies the file's SHA-256 against `model_checksums.txt` before unpickling.

`02_train_model.py` rewrites `model_checksums.txt` after training, so a locally
trained model is trusted automatically. Never add a digest for a model obtained
from an untrusted source; prefer retraining from `voxera_7signs_data.csv`.

## Microphone audio leaves the machine

`recognizer.recognize_google(...)` uploads captured audio to Google's Web Speech
API. The scripts print an explicit `[PRIVACY]` notice at startup. For
deployments where speech must stay local, switch to an offline recognizer
(e.g. Vosk or `recognize_whisper`).

## Dependencies

Dependencies are pinned in `requirements.txt` so upgrades are explicit and
reviewable. Install with `pip install -r requirements.txt` and re-check pins
periodically (`pip-audit`).

## Reporting

Report suspected vulnerabilities by opening a private security advisory on the
GitHub repository.
