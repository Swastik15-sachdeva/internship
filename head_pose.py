"""
head_pose.py — Head pose estimation using MediaPipe Face Mesh
Detects: looking left, right, up, down, or forward
"""

import mediapipe as mp
import numpy as np
from dataclasses import dataclass


# Thresholds (degrees)
YAW_THRESHOLD   = 20   # left/right
PITCH_THRESHOLD = 15   # up/down


@dataclass
class HeadPoseResult:
    direction: str = "forward"   # forward / left / right / up / down
    yaw:   float = 0.0
    pitch: float = 0.0
    roll:  float = 0.0
    detected: bool = False


class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 3D model points for PnP (generic face model)
        self.model_points = np.array([
            (0.0,    0.0,    0.0),    # Nose tip (1)
            (0.0,   -63.6,  -12.5),  # Chin (152)
            (-43.3,  32.7,  -26.0),  # Left eye left corner (33)
            (43.3,   32.7,  -26.0),  # Right eye right corner (263)
            (-28.9, -28.9,  -24.1),  # Left mouth corner (61)
            (28.9,  -28.9,  -24.1),  # Right mouth corner (291)
        ], dtype=np.float64)
        self.landmark_ids = [1, 152, 33, 263, 61, 291]

    def analyze(self, rgb_frame: np.ndarray) -> HeadPoseResult:
        result = HeadPoseResult()
        h, w = rgb_frame.shape[:2]

        mp_result = self.face_mesh.process(rgb_frame)
        if not mp_result.multi_face_landmarks:
            return result

        result.detected = True
        landmarks = mp_result.multi_face_landmarks[0].landmark

        image_points = np.array([
            (landmarks[idx].x * w, landmarks[idx].y * h)
            for idx in self.landmark_ids
        ], dtype=np.float64)

        focal = w
        center = (w / 2, h / 2)
        cam_matrix = np.array([
            [focal, 0,     center[0]],
            [0,     focal, center[1]],
            [0,     0,     1        ]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv_safe_solve(self.model_points, image_points, cam_matrix, dist_coeffs)
        if not success:
            return result

        rot_mat, _ = __import__("cv2").Rodrigues(rvec)
        angles = rotation_matrix_to_angles(rot_mat)
        pitch, yaw, roll = angles
        result.yaw   = float(yaw)
        result.pitch = float(pitch)
        result.roll  = float(roll)

        if yaw < -YAW_THRESHOLD:
            result.direction = "left"
        elif yaw > YAW_THRESHOLD:
            result.direction = "right"
        elif pitch < -PITCH_THRESHOLD:
            result.direction = "down"
        elif pitch > PITCH_THRESHOLD:
            result.direction = "up"
        else:
            result.direction = "forward"

        return result


def cv_safe_solve(model_pts, image_pts, cam_mat, dist):
    import cv2
    return cv2.solvePnP(model_pts, image_pts, cam_mat, dist)


def rotation_matrix_to_angles(R):
    """Convert 3x3 rotation matrix to Euler angles (pitch, yaw, roll) in degrees."""
    import math
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return (
        math.degrees(x),
        math.degrees(y),
        math.degrees(z)
    )
