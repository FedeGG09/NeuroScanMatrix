from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class DocTypeResult:
    document_type: str
    confidence: float
    meta: dict[str, Any] | None = None

class DocumentClassifier:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    def _ensure(self):
        if self.model is not None:
            return
        try:
            from transformers import AutoProcessor, AutoModelForSequenceClassification
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception:
            self.processor = None
            self.model = None

    def predict(self, image: np.ndarray, text_hint: str = "") -> DocTypeResult:
        self._ensure()
        if self.model is not None and self.processor is not None:
            try:
                inputs = self.processor(images=image, text=text_hint, return_tensors="pt")
                outputs = self.model(**inputs)
                logits = outputs.logits.detach().cpu().numpy()[0]
                idx = int(logits.argmax())
                label = self.model.config.id2label.get(idx, f"class_{idx}")
                conf = float(np.exp(logits[idx]) / np.exp(logits).sum())
                return DocTypeResult(label, conf)
            except Exception:
                pass
        # heuristic fallback
        if "dni" in text_hint.lower() or "documento" in text_hint.lower():
            return DocTypeResult("identity_form", 0.58, {"fallback": True})
        return DocTypeResult("generic_form", 0.42, {"fallback": True})
