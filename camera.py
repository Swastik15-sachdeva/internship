"""
camera.py — Webcam capture wrapper using OpenCV
"""

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self.cap.isOpened()

    def read(self):
        """Return (success, BGR frame)."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def capture_images(self, count: int = 15, delay_ms: int = 300):
        """
        Capture `count` frames with delay for registration.
        Returns list of BGR frames.
        """
        frames = []
        if not self.start():
            print("[Camera] Could not open webcam.")
            return frames

        cv2.namedWindow("Registration — press Q to cancel", cv2.WINDOW_NORMAL)
        captured = 0

        while captured < count:
            ok, frame = self.read()
            if not ok:
                break

            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Capturing {captured + 1}/{count} — look at camera",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2
            )
            cv2.imshow("Registration — press Q to cancel", overlay)

            key = cv2.waitKey(delay_ms)
            if key == ord('q') or key == ord('Q'):
                break

            frames.append(frame.copy())
            captured += 1

        cv2.destroyAllWindows()
        self.release()
        return frames

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.release()
