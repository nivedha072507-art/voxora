import time

from voxora_core import SpeechListener

if __name__ == "__main__":
    listener = SpeechListener(language="en-US", calibration_duration=2.0)

    if listener.start():
        print("[INFO] Speech-to-Text active. Speak into your mic. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping speech recognition...")
            listener.stop()
