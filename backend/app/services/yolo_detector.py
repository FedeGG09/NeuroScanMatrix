from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import numpy as np
import cv2


@dataclass
class VisualElement:
    kind: str
    bbox: list[float]
    score: float = 0.0
    meta: dict[str, Any] | None = None


class CheckboxDetector:
    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.model = None

    def _load(self):
        if self.model is not None:
            return self.model
        path = Path(self.weights_path)
        if path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(path))
            except Exception:
                self.model = None
        else:
            self.model = None
        return self.model

    def detect(self, image: np.ndarray) -> list[VisualElement]:
        model = self._load()
        if model is not None:
            try:
                res = model.predict(image, conf=0.25, verbose=False)[0]
                elements = []
                for b in res.boxes:
                    cls = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
                    conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
                    name = model.names.get(cls, "checkbox")
                    xyxy = b.xyxy[0].tolist()
                    elements.append(VisualElement(kind=name, bbox=xyxy, score=conf))
                return elements
            except Exception:
                pass
        return self._heuristic_checkbox_detection(image)

    def _heuristic_checkbox_detection(self, image: np.ndarray) -> list[VisualElement]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # Invertimos para buscar contornos oscuros.
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elements = []
        h, w = gray.shape[:2]

        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            if not (8 <= ww <= 30 and 8 <= hh <= 30):
                continue
            ratio = ww / float(hh)
            if not (0.75 <= ratio <= 1.25):
                continue

            area = cv2.contourArea(cnt)
            box_area = float(max(1, ww * hh))
            fill_ratio = area / box_area
            if not (0.05 <= fill_ratio <= 0.80):
                continue

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.08 * perimeter, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            # Evitar ruido de texto: los checkboxes suelen estar cerca del borde derecho/medio
            # del bloque y aparecer como cuadrados consistentes.
            cx = x + ww / 2.0
            if cx < w * 0.15:
                continue

            elements.append(VisualElement(kind="checkbox", bbox=[x, y, x + ww, y + hh], score=0.35))

        # Deduplicar contornos muy cercanos
        dedup = []
        for e in elements:
            if not any(abs(e.bbox[0] - d.bbox[0]) <= 3 and abs(e.bbox[1] - d.bbox[1]) <= 3 and abs(e.bbox[2] - d.bbox[2]) <= 3 and abs(e.bbox[3] - d.bbox[3]) <= 3 for d in dedup):
                dedup.append(e)
        return dedup


def detect_highlights(image: np.ndarray) -> list[VisualElement]:
    if len(image.shape) != 3:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([15, 40, 130])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500:
            out.append(VisualElement(kind="highlight", bbox=[x, y, x + w, y + h], score=0.5))
    return out
