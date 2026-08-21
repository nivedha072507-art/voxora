"""Camera helpers shared by the capture and translation scripts."""

import cv2


def open_camera(index=0, width=None, height=None, fps=None):
    """Open a webcam, optionally requesting a frame size and frame rate."""
    cap = cv2.VideoCapture(index)

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def camera_resolution(cap):
    """Return the resolution the camera actually settled on."""
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


def read_mirrored_frame(cap):
    """Read a frame, mirror it, and also return its RGB copy for MediaPipe.

    Returns ``(None, None)`` when the camera stops delivering frames.
    """
    ret, frame = cap.read()
    if not ret:
        return None, None

    frame = cv2.flip(frame, 1)
    return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def release_camera(cap):
    """Release the camera and close any OpenCV windows."""
    cap.release()
    cv2.destroyAllWindows()
