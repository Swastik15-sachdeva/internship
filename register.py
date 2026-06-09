"""
register.py — Candidate registration: capture images and build face encoding
"""

import face_recognition
import numpy as np
from modules.camera import Camera
from database.database import save_candidate


def register_candidate(name: str, image_count: int = 15) -> bool:
    """
    Capture images from webcam, compute average face encoding, and store in DB.
    Returns True on success.
    """
    print(f"[Register] Starting registration for '{name}' — capturing {image_count} images.")
    cam = Camera()
    frames = cam.capture_images(count=image_count, delay_ms=350)

    if not frames:
        print("[Register] No frames captured.")
        return False

    encodings = []
    for i, frame in enumerate(frames):
        rgb = frame[:, :, ::-1]  # BGR → RGB
        locs = face_recognition.face_locations(rgb, model="hog")
        if not locs:
            print(f"[Register] Frame {i+1}: no face detected, skipping.")
            continue
        enc = face_recognition.face_encodings(rgb, locs)[0]
        encodings.append(enc)

    if len(encodings) < 3:
        print(f"[Register] Only {len(encodings)} usable frames — registration failed. Try better lighting.")
        return False

    avg_encoding = np.mean(encodings, axis=0)
    success = save_candidate(name, avg_encoding)
    if success:
        print(f"[Register] '{name}' registered successfully ({len(encodings)} frames used).")
    return success
