"""
main.py — AI-Powered Candidate Monitoring and Recruitment Assistance System
Entry point. Run: python main.py

Menu:
  1. Register new candidate
  2. Start monitoring session
  3. List candidates
  4. Exit
"""

import sys
import os
import time
import threading
import cv2
import numpy as np

# Make sub-packages importable from project root
sys.path.insert(0, os.path.dirname(__file__))

from database.database import (
    init_db, load_candidate, list_candidates,
    get_violations, get_risk_score, clear_violations
)
from modules.camera import Camera
from modules.register import register_candidate
from modules.face_recognition_module import FaceRecognizer
from modules.head_pose import HeadPoseDetector
from modules.mouth_detection import MouthDetector
from modules.phone_detection import ObjectDetector
from modules.logger import ViolationLogger
from modules.report_generator import generate_report

# ── Timing constants ──────────────────────────────────────────────────────────
NO_FACE_ALERT_SECONDS    = 10   # Alert if no face detected for this long
LOOK_AWAY_ALERT_SECONDS  = 5    # Alert if looking away for this long
YOLO_EVERY_N_FRAMES      = 5    # Run YOLO every N frames (performance)


class MonitoringSession:
    """
    Core monitoring loop. Runs in a background thread.
    Feeds status updates to the dashboard via a callback.
    """

    def __init__(self, candidate_name: str, candidate_encoding, dashboard=None):
        self.candidate = candidate_name
        self.dashboard = dashboard

        # Detectors
        self.face_recognizer   = FaceRecognizer(candidate_encoding)
        self.head_pose         = HeadPoseDetector()
        self.mouth_detector    = MouthDetector()
        self.object_detector   = ObjectDetector()
        self.logger            = ViolationLogger(candidate_name)

        # State tracking
        self._no_face_since: float = None
        self._look_away_since: float = None
        self._running = False
        self._frame_count = 0

    def start(self):
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        return t

    def stop(self):
        self._running = False

    def _loop(self):
        cam = Camera()
        if not cam.start():
            print("[Monitor] Cannot open webcam.")
            return

        print(f"[Monitor] Session started for '{self.candidate}'. Press Q in preview to stop.")
        cv2.namedWindow("AI Recruiter — Monitoring", cv2.WINDOW_NORMAL)

        while self._running:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.05)
                continue

            self._frame_count += 1
            rgb = frame[:, :, ::-1]
            now = time.time()

            # ── Face recognition ─────────────────────────────
            face_result = self.face_recognizer.analyze(rgb)

            # No-face tracking
            if face_result.face_count == 0:
                if self._no_face_since is None:
                    self._no_face_since = now
                elif now - self._no_face_since >= NO_FACE_ALERT_SECONDS:
                    self.logger.log("No Face", frame)
            else:
                self._no_face_since = None

            # Identity mismatch
            if face_result.identity == "mismatch":
                self.logger.log("Identity Mismatch", frame)

            # Multiple faces
            if face_result.face_count > 1:
                self.logger.log("Multiple Faces", frame)

            # ── Head pose ────────────────────────────────────
            pose_result = self.head_pose.analyze(rgb)
            if pose_result.detected and pose_result.direction != "forward":
                if self._look_away_since is None:
                    self._look_away_since = now
                elif now - self._look_away_since >= LOOK_AWAY_ALERT_SECONDS:
                    self.logger.log("Looking Away", frame)
            else:
                self._look_away_since = None

            # ── Mouth detection ──────────────────────────────
            mouth_result = self.mouth_detector.analyze(rgb)
            if mouth_result.is_talking:
                self.logger.log("Talking", frame)

            # ── Object detection (every N frames) ────────────
            obj_result = None
            if self._frame_count % YOLO_EVERY_N_FRAMES == 0:
                obj_result = self.object_detector.analyze(frame)
                if obj_result.phone_detected:
                    self.logger.log("Phone Detected", frame)

            # ── Annotate preview frame ───────────────────────
            annotated = self._annotate(frame, face_result, pose_result, mouth_result, obj_result)
            cv2.imshow("AI Recruiter — Monitoring", annotated)

            # ── Push status to dashboard ─────────────────────
            if self.dashboard:
                score = get_risk_score(self.candidate)
                violations = self.logger.recent_events(50)
                status = {
                    "identity":   face_result.identity,
                    "face_count": face_result.face_count,
                    "direction":  pose_result.direction if pose_result.detected else "forward",
                    "talking":    mouth_result.is_talking,
                    "phone":      obj_result.phone_detected if obj_result else False,
                    "violations": violations,
                    "score":      score,
                }
                try:
                    self.dashboard.after(0, lambda s=status: self.dashboard.update_status(s))
                    # Log new entries to the dashboard text box
                    recent = self.logger.recent_events(1)
                    if recent:
                        last = recent[-1]
                        entry_text = f"{last['time']}  {last['event']}  (+{last['risk_points']}pts)"
                        self.dashboard.after(0, lambda t=entry_text: self.dashboard.add_log_entry(t))
                except Exception:
                    pass

            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q') or key == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
        print("[Monitor] Session ended.")

    def _annotate(self, frame, face_result, pose_result, mouth_result, obj_result):
        """Draw bounding boxes, labels, and status on the frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Face boxes
        for (top, right, bottom, left) in face_result.locations:
            color = (0, 200, 0) if face_result.identity == "match" else (0, 0, 220)
            cv2.rectangle(out, (left, top), (right, bottom), color, 2)

        # Status overlay
        lines = [
            f"Identity: {face_result.identity}  faces: {face_result.face_count}",
            f"Pose: {pose_result.direction if pose_result.detected else 'no face'}",
            f"Talking: {'yes' if mouth_result.is_talking else 'no'}",
        ]
        if obj_result and obj_result.phone_detected:
            lines.append("PHONE DETECTED")
            # Draw object boxes
            for det in obj_result.detections:
                x1, y1, x2, y2 = det.box
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(out, f"{det.label} {det.confidence:.0%}",
                            (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1)

        for i, line in enumerate(lines):
            cv2.putText(out, line, (10, 25 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(out, line, (10, 25 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1)

        # Risk score badge
        score = get_risk_score(self.candidate)
        badge = f"Risk: {score}"
        cv2.putText(out, badge, (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        color = (40, 180, 40) if score <= 20 else (0, 140, 255) if score <= 50 else (0, 0, 220)
        cv2.putText(out, badge, (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return out


# ── CLI menu ─────────────────────────────────────────────────────────────────

def menu_register():
    name = input("Enter candidate name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return
    count = input("Number of images to capture [15]: ").strip()
    count = int(count) if count.isdigit() else 15
    register_candidate(name, count)


def menu_start_session():
    candidates = list_candidates()
    if not candidates:
        print("No candidates registered. Please register first.")
        return

    print("\nRegistered candidates:")
    for i, (name, created) in enumerate(candidates, 1):
        print(f"  {i}. {name}  (registered {created})")

    choice = input("Enter candidate number or name: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
        name = candidates[int(choice) - 1][0]
    else:
        name = choice

    encoding = load_candidate(name)
    if encoding is None:
        print(f"Candidate '{name}' not found.")
        return

    clear_violations(name)
    print(f"\n[Main] Starting monitoring for '{name}'.")
    print("  → Press Q in the webcam preview to stop the session.")

    use_gui = input("Open recruiter dashboard? [Y/n]: ").strip().lower()

    if use_gui != "n":
        # Import here to avoid loading TK on headless systems
        from gui.dashboard import Dashboard
        dash = Dashboard(candidate_name=name)
        session = MonitoringSession(name, encoding, dashboard=dash)
        thread = session.start()
        dash.mainloop()
        session.stop()
    else:
        session = MonitoringSession(name, encoding)
        thread = session.start()
        thread.join()


def menu_list_candidates():
    candidates = list_candidates()
    if not candidates:
        print("No candidates registered.")
        return
    print("\nRegistered candidates:")
    for name, created in candidates:
        score = get_risk_score(name)
        violations = get_violations(name)
        print(f"  • {name}  (registered {created})  violations: {len(violations)}  risk: {score}")


def main():
    init_db()
    print("=" * 55)
    print("  AI-Powered Recruitment & Proctoring System v1.0")
    print("=" * 55)

    while True:
        print("\n  1. Register new candidate")
        print("  2. Start monitoring session")
        print("  3. List candidates")
        print("  4. Exit")
        choice = input("\nChoice: ").strip()

        if choice == "1":
            menu_register()
        elif choice == "2":
            menu_start_session()
        elif choice == "3":
            menu_list_candidates()
        elif choice == "4":
            print("Goodbye.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
