import cv2
import numpy as np
import pickle
import mediapipe as mp
import speech_recognition as sr
import time
import os
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. GLOBAL STATE & SPEECH RECOGNITION SETUP
# ==========================================
latest_speech_text = "பேசுங்கள்..."  # Default starting text

def speech_callback(recognizer, audio):
    """Background thread callback for handling recognized speech in Tamil."""
    global latest_speech_text
    try:
        # language='ta-IN' configures speech recognition for Tamil
        text = recognizer.recognize_google(audio, language='ta-IN')
        latest_speech_text = text
        print(f"[SPEECH DETECTED]: {text}")
    except sr.UnknownValueError:
        pass  # Ignore unclear audio
    except sr.RequestError as e:
        latest_speech_text = "Speech Service Error"
        print(f"[SPEECH ERROR]: {e}")

def start_speech_recognition():
    """Initializes mic calibration and starts background listening."""
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 300

    try:
        mic = sr.Microphone()
        with mic as source:
            print("[INFO] Calibrating microphone for ambient noise floor...")
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            print("[INFO] Microphone calibrated.")

        stop_listening = recognizer.listen_in_background(mic, speech_callback)
        return stop_listening
    except Exception as e:
        print(f"[ERROR] Microphone initialization failed: {e}")
        return None

# ==========================================
# 2. FEATURE EXTRACTION & MODEL INITIALIZATION
# ==========================================
MODEL_PATH = 'voxera_7signs_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"[INFO] Loaded model: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load model file '{MODEL_PATH}': {e}")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks_sorted(results):
    h1 = np.zeros(63)
    h2 = np.zeros(63)

    if results.multi_hand_landmarks:
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda lm: lm.landmark[0].x)
        
        for idx, hand_landmarks in enumerate(sorted_hands[:2]):
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            mcp = hand_landmarks.landmark[9]
            scale = np.sqrt((mcp.x - base_x)**2 + (mcp.y - base_y)**2 + (mcp.z - base_z)**2)
            if scale == 0:
                scale = 1.0

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([
                    (lm.x - base_x) / scale,
                    (lm.y - base_y) / scale,
                    (lm.z - base_z) / scale
                ])

            if idx == 0:
                h1 = np.array(coords)
            elif idx == 1:
                h2 = np.array(coords)

    return np.concatenate([h1, h2])

# Helper function to render Tamil text on OpenCV frame using Pillow
def draw_tamil_text(img, text, position, font_size=26, color=(255, 200, 0)):
    # Explicit Windows font paths for Tamil rendering
    font_paths = [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\NirmalB.ttf",
        r"C:\Windows\Fonts\latha.ttf",
        "tamil_font.ttf"  # Local folder fallback if custom TTF is present
    ]
    
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==========================================
# 3. MAIN DUAL TRANSLATION LOOP
# ==========================================
def main():
    global missing_hand_counter, latest_speech_text

    stop_listening = start_speech_recognition()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to access camera.")
        if stop_listening:
            stop_listening(wait_for_stop=False)
        return

    prediction_buffer = deque(maxlen=10)
    missing_hand_counter = 0
    displayed_sign = "Waiting..."

    print("[INFO] VOXORA operational. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            missing_hand_counter = 0
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_landmarks_sorted(results)
            probs = model.predict_proba([features])[0]
            max_prob = np.max(probs)
            pred_label = model.classes_[np.argmax(probs)]

            if max_prob > 0.60:
                prediction_buffer.append(pred_label)

            if len(prediction_buffer) > 0:
                most_common, count = Counter(prediction_buffer).most_common(1)[0]
                if count >= 6:
                    displayed_sign = most_common
        else:
            missing_hand_counter += 1
            if missing_hand_counter > 12:
                prediction_buffer.clear()
                displayed_sign = "No Hand Detected"

        # UI Overlay
        # Top Panel: Sign Detection
        cv2.rectangle(frame, (0, 0), (640, 50), (20, 20, 20), -1)
        cv2.putText(frame, f"Sign: {displayed_sign}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Bottom Panel: Speech-To-Text Output (Tamil Supported)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
        
        # Render Tamil text on video stream via PIL
        frame = draw_tamil_text(frame, f"Speech: {latest_speech_text}", (15, h - 42), font_size=26, color=(255, 200, 0))

        cv2.imshow("VOXORA - Sign & Speech Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if stop_listening:
        stop_listening(wait_for_stop=False)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()