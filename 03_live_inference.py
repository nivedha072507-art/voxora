import cv2

from voxora_core import (
    SignSmoother,
    create_hand_detector,
    draw_hand_landmarks,
    extract_landmarks_sorted,
    load_model,
    open_camera,
    predict_sign,
    read_mirrored_frame,
    release_camera,
)

model = load_model(exit_on_error=True)
hands = create_hand_detector(detection_confidence=0.5)


def main():
    cap = open_camera()
    smoother = SignSmoother()

    while cap.isOpened():
        frame, rgb_frame = read_mirrored_frame(cap)
        if frame is None:
            break

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            draw_hand_landmarks(frame, results)
            label, confidence = predict_sign(
                model, extract_landmarks_sorted(results)
            )
            displayed_sign = smoother.update(label, confidence)
        else:
            displayed_sign = smoother.update_missing()

        # UI Header
        cv2.rectangle(frame, (0, 0), (450, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Voxera Sign: {displayed_sign}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Voxera Live Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_camera(cap)


if __name__ == "__main__":
    main()
