"""
phone_detection.py — Object detection using YOLOv8
Detects: mobile phone, book, laptop, extra person
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


# COCO class IDs of interest
COCO_PHONE   = 67   # cell phone
COCO_BOOK    = 73   # book
COCO_LAPTOP  = 63   # laptop
COCO_PERSON  = 0    # person

CONFIDENCE_THRESHOLD = 0.45

TARGET_CLASSES = {
    COCO_PHONE:  "phone",
    COCO_BOOK:   "book",
    COCO_LAPTOP: "laptop",
    COCO_PERSON: "person",
}


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple   # (x1, y1, x2, y2)


@dataclass
class ObjectResult:
    phone_detected:  bool = False
    book_detected:   bool = False
    laptop_detected: bool = False
    person_count:    int  = 0
    detections: List[Detection] = field(default_factory=list)


class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self._model = None
        self._model_path = model_path

    def _load_model(self):
        """Lazy-load model on first use to save startup time."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            print("[ObjectDetector] YOLOv8 model loaded.")

    def analyze(self, bgr_frame: np.ndarray) -> ObjectResult:
        self._load_model()
        result = ObjectResult()

        yolo_results = self._model.predict(
            bgr_frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            classes=list(TARGET_CLASSES.keys())
        )

        if not yolo_results:
            return result

        boxes = yolo_results[0].boxes
        if boxes is None:
            return result

        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = TARGET_CLASSES.get(cls_id, "unknown")

            det = Detection(label=label, confidence=conf, box=(x1, y1, x2, y2))
            result.detections.append(det)

            if cls_id == COCO_PHONE:
                result.phone_detected = True
            elif cls_id == COCO_BOOK:
                result.book_detected = True
            elif cls_id == COCO_LAPTOP:
                result.laptop_detected = True
            elif cls_id == COCO_PERSON:
                result.person_count += 1

        return result
