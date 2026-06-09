"""
face_recognition_module.py — Identity verification and face counting
"""

import face_recognition
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


TOLERANCE = 0.5  # Lower = stricter match


@dataclass
class FaceResult:
    face_count: int = 0
    identity: str = "unknown"      # "match", "mismatch", "unknown", "no_face"
    match_distance: float = 1.0
    locations: list = field(default_factory=list)


class FaceRecognizer:
    def __init__(self, candidate_encoding: Optional[np.ndarray] = None):
        self.candidate_encoding = candidate_encoding

    def set_encoding(self, encoding: np.ndarray):
        self.candidate_encoding = encoding

    def analyze(self, rgb_frame: np.ndarray) -> FaceResult:
        """
        Analyze a single RGB frame.
        Returns FaceResult with count, identity status, and face locations.
        """
        result = FaceResult()

        locations = face_recognition.face_locations(rgb_frame, model="hog")
        result.face_count = len(locations)
        result.locations = locations

        if not locations:
            result.identity = "no_face"
            return result

        if self.candidate_encoding is None:
            result.identity = "unknown"
            return result

        encodings = face_recognition.face_encodings(rgb_frame, locations)
        if not encodings:
            result.identity = "unknown"
            return result

        # Compare first (largest) face to registered candidate
        distances = face_recognition.face_distance([self.candidate_encoding], encodings[0])
        dist = float(distances[0])
        result.match_distance = dist

        if dist <= TOLERANCE:
            result.identity = "match"
        else:
            result.identity = "mismatch"

        return result
