import cv2

from voxora_core import (
    SIGN_TAMIL,
    SignSmoother,
    SpeechListener,
    camera_resolution,
    create_hand_detector,
    draw_hand_landmarks,
    draw_tamil_text,
    extract_landmarks_sorted,
    load_model,
    open_camera,
    predict_sign,
    read_mirrored_frame,
    release_camera,
)

WINDOW_NAME = "VOXORA - Sign & Speech Translator"

LANGUAGES = {
    ord('a'): ("ta-IN", "TAMIL", "பேசுங்கள்..."),
    ord('s'): ("en-IN", "ENGLISH", "Speak..."),
}

model = load_model(exit_on_error=True)
hands = create_hand_detector(detection_confidence=0.55)


def main():
    language_code, language_name, prompt = LANGUAGES[ord('a')]

    listener = SpeechListener(language=language_code, initial_text=prompt)
    listener.start()

    cap = open_camera(width=1280, height=720, fps=30)

    actual_w, actual_h = camera_resolution(cap)
    print(f"[INFO] Camera resolution: {actual_w} x {actual_h}")

    if not cap.isOpened():
        print("[ERROR] Unable to access camera.")
        listener.stop()
        return

    smoother = SignSmoother()

    print()
    print("================================")
    print("        VOXORA STARTED")
    print("================================")
    print()
    print("A = Tamil")
    print("S = English")
    print("Q = Exit")
    print()
    print(f"Current Language: {language_name}")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 1050)

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

        h, w, _ = frame.shape

        # ----------------------------------
        # TOP HEADER
        # ----------------------------------
        TOP_H = 92
        cv2.rectangle(frame, (0, 0), (w, TOP_H), (18, 18, 18), -1)

        cv2.putText(frame, "VOXORA", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        cv2.putText(frame, f"Language: {language_name}", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

        cv2.putText(frame, f"Sign: {displayed_sign}", (w - 520, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        frame = draw_tamil_text(
            frame,
            SIGN_TAMIL.get(displayed_sign, displayed_sign),
            (w - 310, 45),
            font_size=25,
            color=(0, 255, 255),
            max_width=280,
            max_lines=1,
            bold=True,
        )

        # ----------------------------------
        # SPEECH OUTPUT PANEL
        # ----------------------------------
        SPEECH_PANEL_HEIGHT = 190
        cv2.rectangle(frame, (0, h - SPEECH_PANEL_HEIGHT), (w, h), (18, 18, 18), -1)

        cv2.putText(frame, "SPEECH", (24, h - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        frame = draw_tamil_text(
            frame,
            listener.text,
            (24, h - 118),
            font_size=31,
            color=(255, 255, 255),
            max_width=w - 48,
            max_lines=2,
            bold=False,
        )

        cv2.putText(frame, listener.status, (24, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        for text, x in (("A  Tamil", w - 290), ("S  English", w - 190), ("Q  Exit", w - 82)):
            cv2.putText(frame, text, (x, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        cv2.line(frame, (0, h - SPEECH_PANEL_HEIGHT), (w, h - SPEECH_PANEL_HEIGHT),
                 (70, 70, 70), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in LANGUAGES:
            language_code, language_name, prompt = LANGUAGES[key]
            listener.set_language(
                language_code,
                initial_text=prompt,
                status=f"Ready - {language_name.title()}",
            )
            print(f"[LANGUAGE] {language_name.title()} selected")
        elif key == ord('q'):
            break

    listener.stop()
    release_camera(cap)
    hands.close()

    print()
    print("VOXORA stopped.")


if __name__ == "__main__":
    main()
