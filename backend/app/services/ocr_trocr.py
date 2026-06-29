from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class TrOCRResult:
    text: str
    confidence: float = 0.0

class TrOCREngine:
    def __init__(self, printed_model: str, handwritten_model: str, device: str = "cpu"):
        self.printed_model = printed_model
        self.handwritten_model = handwritten_model
        self.device = device
        self._printed = None
        self._handwritten = None

    def _load(self, model_name: str):
        try:
            from transformers import pipeline
            return pipeline(
                "image-to-text",
                model=model_name,
                device=-1 if self.device == "cpu" else 0,
            )
        except Exception:
            return None

    def _ensure(self, handwritten: bool = False):
        if handwritten:
            if self._handwritten is None:
                self._handwritten = self._load(self.handwritten_model)
            return self._handwritten
        if self._printed is None:
            self._printed = self._load(self.printed_model)
        return self._printed

    def predict(self, image: np.ndarray, handwritten: bool = False) -> TrOCRResult:
        pipe = self._ensure(handwritten=handwritten)
        if pipe is None:
            return TrOCRResult(text="")
        try:
            out = pipe(image)
            if isinstance(out, list) and out:
                text = out[0].get("generated_text", "") if isinstance(out[0], dict) else str(out[0])
            elif isinstance(out, dict):
                text = out.get("generated_text", "")
            else:
                text = str(out)
            return TrOCRResult(text=text.strip(), confidence=0.0)
        except Exception:
            return TrOCRResult(text="")
