import cv2
import numpy as np
import mediapipe as mp
import speech_recognition as sr
import time
import os
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

from model_loader import ModelIntegrityError, load_model

# ==========================================
# 1. GLOBAL STATE & SPEECH RECOGNITION
# ==========================================

latest_speech_text = "பேசுங்கள்..."
speech_status = "Ready"
current_language = "ta-IN"
language_name = "TAMIL"

# ------------------------------------------
# SIGN LANGUAGE -> TAMIL MAPPING
# ------------------------------------------

SIGN_TAMIL = {
    "YES": "ஆம்",
    "NO": "இல்லை",
    "WHAT": "என்ன",
    "WHERE": "எங்கே",
    "HELP": "உதவி",
    "WHEN": "எப்போது",
    "STOP": "நிறுத்து"
}


# ------------------------------------------
# LANGUAGE SETTINGS
# ------------------------------------------

def set_language(key):
    global current_language
    global language_name
    global latest_speech_text
    global speech_status

    if key == ord('a'):
        current_language = "ta-IN"
        language_name = "TAMIL"
        latest_speech_text = "பேசுங்கள்..."
        speech_status = "Ready - Tamil"
        print("[LANGUAGE] Tamil selected")

    elif key == ord('s'):
        current_language = "en-IN"
        language_name = "ENGLISH"
        latest_speech_text = "Speak..."
        speech_status = "Ready - English"
        print("[LANGUAGE] English selected")


# ------------------------------------------
# SPEECH CALLBACK
# ------------------------------------------

def speech_callback(recognizer, audio):

    global latest_speech_text
    global current_language
    global speech_status

    try:

        text = recognizer.recognize_google(
            audio,
            language=current_language
        )

        if text.strip():
            latest_speech_text = text
            speech_status = "Listening"

            print(
                f"[SPEECH - {current_language}]: {text}"
            )

    except sr.UnknownValueError:

        # Speech was heard but could not be understood.
        # Keep the previous text instead of showing an error.
        speech_status = "Could not understand"

    except sr.RequestError as e:

        # Internet/Google recognition service may temporarily fail.
        # Do not replace the user's last valid speech with an error.
        speech_status = "Speech service unavailable"
        print(f"[SPEECH SERVICE WARNING]: {e}")

    except Exception as e:

        speech_status = "Speech retrying"
        print(f"[SPEECH WARNING]: {e}")


# ------------------------------------------
# START SPEECH RECOGNITION
# ------------------------------------------

def start_speech_recognition():

    print(
        "[PRIVACY] Captured microphone audio is uploaded to Google's "
        "Web Speech API for transcription."
    )

    recognizer = sr.Recognizer()

    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 250
    recognizer.pause_threshold = 1.0
    recognizer.phrase_threshold = 0.15
    recognizer.non_speaking_duration = 0.5

    try:

        # Prefer the Windows Microphone Array if available.
        # This avoids accidentally selecting a speaker/stereo-mix device.
        device_index = None

        try:
            available = sr.Microphone.list_microphone_names()

            for i, name in enumerate(available):
                name_lower = name.lower()
                if (
                    "microphone array" in name_lower
                    or ("microphone" in name_lower and "realtek" in name_lower)
                ):
                    device_index = i
                    print(f"[INFO] Using microphone device {i}: {name}")
                    break

            if device_index is None:
                print("[INFO] Using Windows default microphone.")

        except Exception as e:
            print(f"[INFO] Could not enumerate microphones: {e}")

        mic = sr.Microphone(device_index=device_index)

        with mic as source:

            print(
                "[INFO] Calibrating microphone "
                "for ambient noise floor..."
            )

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1.0
            )

            # Prevent a noisy room from making the microphone too insensitive.
            if recognizer.energy_threshold > 500:
                recognizer.energy_threshold = 500

            # Keep a sensible lower bound for a normal laptop microphone.
            if recognizer.energy_threshold < 120:
                recognizer.energy_threshold = 120

            print(
                f"[INFO] Microphone calibrated. "
                f"Energy threshold: {recognizer.energy_threshold:.0f}"
            )

        stop_listening = recognizer.listen_in_background(
            mic,
            speech_callback,
            phrase_time_limit=10
        )

        return stop_listening

    except Exception as e:

        print(
            f"[ERROR] Microphone initialization failed: {e}"
        )

        return None


# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================

try:

    model = load_model()

    print("[INFO] Loaded model (SHA-256 verified).")

except ModelIntegrityError as e:

    print(f"[ERROR] Model integrity check failed: {e}")

    raise SystemExit(1)


# ==========================================
# 3. MEDIAPIPE HAND DETECTION
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)


# ==========================================
# 4. LANDMARK FEATURE EXTRACTION
# ==========================================

def extract_landmarks_sorted(results):

    h1 = np.zeros(63)
    h2 = np.zeros(63)

    if results.multi_hand_landmarks:

        sorted_hands = sorted(
            results.multi_hand_landmarks,
            key=lambda lm: lm.landmark[0].x
        )

        for idx, hand_landmarks in enumerate(
            sorted_hands[:2]
        ):

            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            mcp = hand_landmarks.landmark[9]

            scale = np.sqrt(
                (mcp.x - base_x) ** 2 +
                (mcp.y - base_y) ** 2 +
                (mcp.z - base_z) ** 2
            )

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


# ==========================================
# 5. TAMIL TEXT RENDERING
# ==========================================

def draw_tamil_text(
    img,
    text,
    position,
    font_size=24,
    color=(255, 200, 0),
    max_width=650,
    max_lines=3,
    bold=False
):
    """Render Tamil/English text using the modern Windows Tamil font."""

    # Your Windows installation has Nirmala as a TrueType Collection (.ttc).
    # Use that first so VOXORA matches the Tamil font already installed on the laptop.
    if bold:
        font_paths = [
            r"C:\Windows\Fonts\Nirmala.ttc",
            r"C:\Windows\Fonts\NirmalB.ttf",
            r"C:\Windows\Fonts\NirmalaUIB.ttf",
            r"C:\Windows\Fonts\NirmalaUI.ttf",
            r"C:\Windows\Fonts\Nirmala.ttf"
        ]
    else:
        font_paths = [
            r"C:\Windows\Fonts\Nirmala.ttc",
            r"C:\Windows\Fonts\NirmalaUI.ttf",
            r"C:\Windows\Fonts\Nirmala.ttf",
            r"C:\Windows\Fonts\NirmalaUIB.ttf"
        ]

    font = None

    for path in font_paths:
        if os.path.exists(path):
            try:
                if path.lower().endswith(".ttc"):
                    font = ImageFont.truetype(path, font_size, index=0)
                else:
                    font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                pass

    if font is None:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    draw = ImageDraw.Draw(img_pil)

    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = word if not current_line else current_line + " " + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines[-1] and not lines[-1].endswith("..."):
            lines[-1] += "..."

    x, y = position
    line_spacing = font_size + 14

    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=color
        )
        y += line_spacing

    return cv2.cvtColor(
        np.array(img_pil),
        cv2.COLOR_RGB2BGR
    )


# ==========================================
# 6. MAIN VOXORA LOOP
# ==========================================

def main():

    global latest_speech_text
    global current_language
    global language_name

    # Start microphone
    stop_listening = start_speech_recognition()

    # Start camera
    cap = cv2.VideoCapture(0)

    # Request a larger camera frame for better hand detection and a
    # larger prototype display.
    # Request a larger 16:9 camera frame.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Read back the actual camera size (some webcams choose the nearest mode).
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Camera resolution: {actual_w} x {actual_h}")

    if not cap.isOpened():

        print(
            "[ERROR] Unable to access camera."
        )

        if stop_listening:

            stop_listening(
                wait_for_stop=False
            )

        return

    prediction_buffer = deque(
        maxlen=10
    )

    missing_hand_counter = 0

    displayed_sign = "Waiting..."

    print()
    print("================================")
    print("        VOXORA STARTED")
    print("================================")
    print()
    print("A = Tamil")
    print("S = English")
    print("Q = Exit")
    print()
    print(
        f"Current Language: {language_name}"
    )
    print()

    # --------------------------------------
    # MAIN LOOP
    # --------------------------------------

    WINDOW_NAME = "VOXORA - Sign & Speech Translator"

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    # Make the application window larger.
    cv2.resizeWindow(
        WINDOW_NAME,
        1400,
        1050
    )

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(
            rgb_frame
        )

        # ==================================
        # HAND DETECTION
        # ==================================

        if results.multi_hand_landmarks:

            missing_hand_counter = 0

            for hand_landmarks in (
                results.multi_hand_landmarks
            ):

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            # Extract features
            features = extract_landmarks_sorted(
                results
            )

            # Model probabilities
            probs = model.predict_proba(
                [features]
            )[0]

            max_prob = np.max(probs)

            pred_label = model.classes_[
                np.argmax(probs)
            ]

            # Confidence threshold
            if max_prob > 0.60:

                prediction_buffer.append(
                    pred_label
                )

            # Majority voting
            if len(prediction_buffer) > 0:

                most_common, count = (
                    Counter(
                        prediction_buffer
                    ).most_common(1)[0]
                )

                if count >= 6:

                    displayed_sign = (
                        most_common
                    )

        else:

            missing_hand_counter += 1

            if missing_hand_counter > 12:

                prediction_buffer.clear()

                displayed_sign = (
                    "No Hand Detected"
                )

        # ==================================
        # UI
        # ==================================

        h, w, _ = frame.shape

        # ----------------------------------
        # TOP HEADER
        # ----------------------------------
        TOP_H = 92

        cv2.rectangle(
            frame,
            (0, 0),
            (w, TOP_H),
            (18, 18, 18),
            -1
        )

        # VOXORA
        cv2.putText(
            frame,
            "VOXORA",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Language: {language_name}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2
        )

        # Bilingual sign output
        tamil_sign = SIGN_TAMIL.get(
            displayed_sign,
            displayed_sign
        )

        cv2.putText(
            frame,
            f"Sign: {displayed_sign}",
            (w - 520, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        frame = draw_tamil_text(
            frame,
            tamil_sign,
            (w - 310, 45),
            font_size=25,
            color=(0, 255, 255),
            max_width=280,
            max_lines=1,
            bold=True
        )

        # ----------------------------------
        # CLEAN SPEECH OUTPUT PANEL
        # ----------------------------------

        SPEECH_PANEL_HEIGHT = 190

        # Main speech panel
        cv2.rectangle(
            frame,
            (0, h - SPEECH_PANEL_HEIGHT),
            (w, h),
            (18, 18, 18),
            -1
        )

        # Small title
        cv2.putText(
            frame,
            "SPEECH",
            (24, h - 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1
        )

        # Tamil/English speech text.
        # Give it a wide area and enough vertical room for 2 lines.
        frame = draw_tamil_text(
            frame,
            latest_speech_text,
            (24, h - 118),
            font_size=31,
            color=(255, 255, 255),
            max_width=w - 48,
            max_lines=2,
            bold=False
        )

        # Status and controls stay at the bottom.
        cv2.putText(
            frame,
            speech_status,
            (24, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 150, 150),
            1
        )

        cv2.putText(
            frame,
            "A  Tamil",
            (w - 290, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            "S  English",
            (w - 190, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            "Q  Exit",
            (w - 82, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )

        # Thin separator above speech area
        cv2.line(
            frame,
            (0, h - SPEECH_PANEL_HEIGHT),
            (w, h - SPEECH_PANEL_HEIGHT),
            (70, 70, 70),
            2
        )

        # ==================================
        # SHOW WINDOW
        # ==================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # ==================================
        # KEYBOARD CONTROL
        # ==================================

        key = cv2.waitKey(1) & 0xFF

        # Tamil
        if key == ord('a'):

            set_language(
                ord('a')
            )

        # English
        elif key == ord('s'):

            set_language(
                ord('s')
            )

        # Quit
        elif key == ord('q'):

            break

    # ======================================
    # CLEANUP
    # ======================================

    if stop_listening:

        stop_listening(
            wait_for_stop=False
        )

    cap.release()

    cv2.destroyAllWindows()

    hands.close()

    print()
    print("VOXORA stopped.")


# ==========================================
# PROGRAM START
# ==========================================

if __name__ == "__main__":

    main()