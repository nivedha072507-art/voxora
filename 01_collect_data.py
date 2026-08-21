
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

SIGNS = ["YES", "NO", "WHEN", "WHERE", "STOP", "HELP", "WHAT"]
SAMPLES_PER_SIGN = 150

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

CSV_FILE = 'voxera_7signs_data.csv'


def ensure_dataset_header():
    """Create the dataset with its header only when it does not exist yet."""

    if os.path.exists(CSV_FILE):
        return

    with open(CSV_FILE, mode='x', newline='') as f:
        writer = csv.writer(f)
        header = [f'coord_{i}' for i in range(126)] + ['label']
        writer.writerow(header)

def extract_landmarks_sorted(results):
    h1 = np.zeros(63)
    h2 = np.zeros(63)

    if results.multi_hand_landmarks:
        # Sort hands strictly by X position (Left-to-Right in image)
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda lm: lm.landmark[0].x)
        
        for idx, hand_landmarks in enumerate(sorted_hands[:2]):
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            # Scale normalization relative to wrist-to-middle MCP distance
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

def collect_data():
    ensure_dataset_header()
    cap = cv2.VideoCapture(0)
    
    for sign in SIGNS:
        print(f"\n==========================================")
        print(f" Get Ready to Record Sign: '{sign}'")
        print(f" Press 's' to start recording {SAMPLES_PER_SIGN} samples.")
        print(f" Perform slight movements/rotations while recording.")
        print(f"==========================================\n")
        
        recording = False
        count = 0

        while cap.isOpened() and count < SAMPLES_PER_SIGN:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and recording:
                features = extract_landmarks_sorted(results)
                row = list(features) + [sign]

                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                count += 1
                cv2.putText(frame, f"Saved: {count}/{SAMPLES_PER_SIGN}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Target: {sign}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            if not recording:
                cv2.putText(frame, "Press 'S' to Start Recording", (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Voxera Precision Data Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                recording = True
                time.sleep(0.3)
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
