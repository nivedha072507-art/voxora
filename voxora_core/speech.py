"""Background speech recognition shared by the speech and translator scripts."""

import speech_recognition as sr

PREFERRED_MIC_KEYWORDS = ("microphone array", "realtek")


def _pick_microphone_index(keywords=PREFERRED_MIC_KEYWORDS):
    """Return the index of a preferred input device, or None for the default.

    Picking an explicit microphone avoids grabbing a loopback/stereo-mix device
    on machines where that is the system default.
    """
    if not keywords:
        return None

    try:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in keywords):
                print(f"[INFO] Using microphone device {index}: {name}")
                return index
    except Exception as e:
        print(f"[INFO] Could not enumerate microphones: {e}")
        return None

    print("[INFO] Using default microphone.")
    return None


class SpeechListener:
    """Listens on a background thread and exposes the latest recognised text."""

    def __init__(
        self,
        language="en-IN",
        initial_text="",
        energy_threshold=250,
        min_energy_threshold=120,
        max_energy_threshold=500,
        calibration_duration=1.0,
        phrase_time_limit=10,
        preferred_mic_keywords=PREFERRED_MIC_KEYWORDS,
        on_text=None,
    ):
        self.language = language
        self.text = initial_text
        self.status = "Ready"
        self.energy_threshold = energy_threshold
        self.min_energy_threshold = min_energy_threshold
        self.max_energy_threshold = max_energy_threshold
        self.calibration_duration = calibration_duration
        self.phrase_time_limit = phrase_time_limit
        self.preferred_mic_keywords = preferred_mic_keywords
        self.on_text = on_text
        self._stop_listening = None

    def _build_recognizer(self):
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = self.energy_threshold
        recognizer.pause_threshold = 1.0
        recognizer.phrase_threshold = 0.15
        recognizer.non_speaking_duration = 0.5
        return recognizer

    def _clamp_energy_threshold(self, recognizer):
        if self.max_energy_threshold is not None:
            recognizer.energy_threshold = min(
                recognizer.energy_threshold, self.max_energy_threshold
            )
        if self.min_energy_threshold is not None:
            recognizer.energy_threshold = max(
                recognizer.energy_threshold, self.min_energy_threshold
            )

    def _callback(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio, language=self.language)
            if text.strip():
                self.text = text
                self.status = "Listening"
                print(f"[SPEECH - {self.language}]: {text}")
                if self.on_text:
                    self.on_text(text)
        except sr.UnknownValueError:
            # Speech was heard but not understood; keep the previous text.
            self.status = "Could not understand"
        except sr.RequestError as e:
            self.status = "Speech service unavailable"
            print(f"[SPEECH SERVICE WARNING]: {e}")
        except Exception as e:
            self.status = "Speech retrying"
            print(f"[SPEECH WARNING]: {e}")

    def start(self):
        """Calibrate the microphone and start listening in the background."""
        recognizer = self._build_recognizer()

        try:
            mic = sr.Microphone(
                device_index=_pick_microphone_index(self.preferred_mic_keywords)
            )

            with mic as source:
                print("[INFO] Calibrating microphone for ambient noise floor...")
                recognizer.adjust_for_ambient_noise(
                    source, duration=self.calibration_duration
                )
                self._clamp_energy_threshold(recognizer)
                print(
                    "[INFO] Microphone calibrated. Energy threshold: "
                    f"{recognizer.energy_threshold:.0f}"
                )

            self._stop_listening = recognizer.listen_in_background(
                mic,
                self._callback,
                phrase_time_limit=self.phrase_time_limit,
            )
            return True
        except Exception as e:
            print(f"[ERROR] Microphone initialization failed: {e}")
            self.status = "Microphone unavailable"
            return False

    def set_language(self, language, initial_text="", status=None):
        """Switch recognition language and reset the displayed text."""
        self.language = language
        self.text = initial_text
        self.status = status or f"Ready - {language}"

    def stop(self, wait_for_stop=False):
        """Stop the background listening thread if it is running."""
        if self._stop_listening:
            self._stop_listening(wait_for_stop=wait_for_stop)
            self._stop_listening = None
