import cv2
import numpy as np
import pickle
import mediapipe as mp
from collections import deque, Counter

MODEL_PATH = 'voxera_7signs_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 10-frame buffer for smooth predictions
prediction_buffer = deque(maxlen=10)
# Grace counter to prevent "No Hand Detected" flashing during transitions
missing_hand_counter = 0

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

def main():
    global missing_hand_counter
    cap = cv2.VideoCapture(0)
    displayed_sign = "Waiting..."

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

            # High confidence threshold eliminates false cross-predictions
            if max_prob > 0.60:
                prediction_buffer.append(pred_label)

            if len(prediction_buffer) > 0:
                most_common, count = Counter(prediction_buffer).most_common(1)[0]
                # Requires at least 6 out of 10 consistent frames to change prediction
                if count >= 6:
                    displayed_sign = most_common
        else:
            missing_hand_counter += 1
            # Only switch to "No Hand Detected" after 12 consecutive frames without hands
            if missing_hand_counter > 12:
                prediction_buffer.clear()
                displayed_sign = "No Hand Detected"

        # UI Header
        cv2.rectangle(frame, (0, 0), (450, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Voxera Sign: {displayed_sign}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Voxera Live Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
