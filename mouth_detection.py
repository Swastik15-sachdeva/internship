"""
mouth_detection.py — Mouth open / talking detection using MediaPipe
Uses the Mouth Aspect Ratio (MAR) from lip landmarks.
"""

import mediapipe as mp
import numpy as np
from dataclasses import dataclass


MAR_THRESHOLD = 0.04      # Ratio above which mouth is considered open
TALKING_FRAMES = 8        # Consecutive open frames = "talking"


@dataclass
class MouthResult:
    is_open: bool = False
    is_talking: bool = False
    mar: float = 0.0
    detected: bool = False


class MouthDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Upper lip: 13, Lower lip: 14, Left corner: 61, Right corner: 291
        # Upper inner: 82, Lower inner: 87  (vertical openness)
        self._open_streak = 0

    def analyze(self, rgb_frame: np.ndarray) -> MouthResult:
        result = MouthResult()
        h, w = rgb_frame.shape[:2]

        mp_result = self.face_mesh.process(rgb_frame)
        if not mp_result.multi_face_landmarks:
            self._open_streak = 0
            return result

        result.detected = True
        lm = mp_result.multi_face_landmarks[0].landmark

        # Vertical distance between upper and lower inner lip points
        upper = np.array([lm[82].x * w,  lm[82].y * h])
        lower = np.array([lm[87].x * w,  lm[87].y * h])
        left  = np.array([lm[61].x * w,  lm[61].y * h])
        right = np.array([lm[291].x * w, lm[291].y * h])

        vertical   = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left  - right)

        mar = vertical / (horizontal + 1e-6)
        result.mar = round(float(mar), 4)
        result.is_open = mar > MAR_THRESHOLD

        if result.is_open:
            self._open_streak += 1
        else:
            self._open_streak = max(0, self._open_streak - 1)

        result.is_talking = self._open_streak >= TALKING_FRAMES
        return result

    def reset(self):
        self._open_streak = 0
