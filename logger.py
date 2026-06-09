"""
logger.py — Violation event logger with screenshot capture and cooldown logic
"""

import cv2
import os
import time
from datetime import datetime
from database.database import log_violation


SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Risk score per event type
RISK_SCORES = {
    "Looking Away":       5,
    "Talking":            5,
    "No Face":           10,
    "Multiple Faces":    20,
    "Phone Detected":    30,
    "Identity Mismatch": 50,
    "Mouth Open":         3,
}

# Minimum seconds between logging the same event (cooldown)
COOLDOWNS = {
    "Looking Away":    8,
    "Talking":        10,
    "No Face":        10,
    "Multiple Faces":  5,
    "Phone Detected":  8,
    "Identity Mismatch": 15,
    "Mouth Open":     12,
}


class ViolationLogger:
    def __init__(self, candidate_name: str):
        self.candidate = candidate_name
        self._last_logged: dict = {}    # event -> last epoch time
        self.in_memory: list = []       # list of dicts for dashboard

    def log(self, event: str, frame=None) -> bool:
        """
        Log an event if cooldown has elapsed.
        Saves a screenshot if frame is provided.
        Returns True if event was logged.
        """
        now = time.time()
        cooldown = COOLDOWNS.get(event, 8)
        last = self._last_logged.get(event, 0)

        if now - last < cooldown:
            return False

        self._last_logged[event] = now

        # Save screenshot
        img_path = None
        if frame is not None:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{self.candidate}_{event.replace(' ', '_')}_{ts_str}.jpg"
            img_path = os.path.join(SCREENSHOTS_DIR, fname)
            cv2.imwrite(img_path, frame)

        pts = RISK_SCORES.get(event, 0)
        log_violation(self.candidate, event, img_path, pts)

        entry = {
            "event": event,
            "time": datetime.now().strftime("%H:%M:%S"),
            "risk_points": pts,
            "image_path": img_path,
        }
        self.in_memory.append(entry)
        print(f"[Logger] {entry['time']} — {event} ({pts}pts)")
        return True

    def recent_events(self, n: int = 10):
        """Return last n events for the dashboard."""
        return self.in_memory[-n:]

    def reset(self):
        self._last_logged.clear()
        self.in_memory.clear()
