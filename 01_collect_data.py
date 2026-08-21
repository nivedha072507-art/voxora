import csv
import time

import cv2

from voxora_core import (
    DATA_CSV_PATH,
    FEATURE_COUNT,
    SIGNS,
    create_hand_detector,
    extract_landmarks_sorted,
    open_camera,
    read_mirrored_frame,
    release_camera,
)

SAMPLES_PER_SIGN = 150

hands = create_hand_detector(detection_confidence=0.6)


def write_csv_header(path=DATA_CSV_PATH):
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"coord_{i}" for i in range(FEATURE_COUNT)] + ["label"])


def append_sample(features, label, path=DATA_CSV_PATH):
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(features) + [label])


def collect_data():
    cap = open_camera()

    for sign in SIGNS:
        print("\n==========================================")
        print(f" Get Ready to Record Sign: '{sign}'")
        print(f" Press 's' to start recording {SAMPLES_PER_SIGN} samples.")
        print(" Perform slight movements/rotations while recording.")
        print("==========================================\n")

        recording = False
        count = 0

        while cap.isOpened() and count < SAMPLES_PER_SIGN:
            frame, rgb_frame = read_mirrored_frame(cap)
            if frame is None:
                break

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and recording:
                append_sample(extract_landmarks_sorted(results), sign)
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
            if key == ord("s"):
                recording = True
                time.sleep(0.3)
            elif key == ord("q"):
                break

    release_camera(cap)


if __name__ == "__main__":
    write_csv_header()
    collect_data()
