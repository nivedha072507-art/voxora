import speech_recognition as sr
import time

# Global variable to store recognized text
latest_speech_text = ""

def speech_callback(recognizer, audio):
    """Callback function triggered when audio capture completes in background."""
    global latest_speech_text
    try:
        # Use Google Web Speech API for recognition
        text = recognizer.recognize_google(audio)
        latest_speech_text = text
        print(f"[SPEECH]: {text}")
    except sr.UnknownValueError:
        # Audio was unintelligible
        pass
    except sr.RequestError as e:
        print(f"[SPEECH ERROR]: Could not request results; {e}")

def start_speech_recognition():
    """Initializes mic and starts background listening thread."""
    recognizer = sr.Recognizer()
    
    # Adjust dynamic energy thresholds for line/ambient noise
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 300

    try:
        mic = sr.Microphone()
        with mic as source:
            print("[INFO] Calibrating microphone for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("[INFO] Microphone calibrated. Listening in background...")

        # listen_in_background runs on a non-blocking background thread
        stop_listening = recognizer.listen_in_background(mic, speech_callback)
        return stop_listening
    except Exception as e:
        print(f"[ERROR] Could not start speech recognition: {e}")
        return None

if __name__ == "__main__":
    stop_fn = start_speech_recognition()
    
    if stop_fn:
        print("[INFO] Speech-to-Text active. Speak into your mic. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping speech recognition...")
            stop_fn(wait_for_stop=False)