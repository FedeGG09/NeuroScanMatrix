from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import re

@dataclass
class LayoutField:
    key: str
    value: str | None
    confidence: float
    source: str = "layoutlmv3"
    bbox: list[float] | None = None
    page: int | None = None
    ambiguous: bool = False
    candidates: list[str] | None = None

class LayoutLMv3Extractor:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    def _ensure(self):
        if self.model is not None:
            return
        try:
            from transformers import AutoProcessor, AutoModelForTokenClassification
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        except Exception:
            self.processor = None
            self.model = None

    def extract_from_ocr(self, tokens: list[dict[str, Any]], image=None) -> dict[str, LayoutField]:
        self._ensure()
        text = " ".join(t.get("text", "") for t in tokens)
        # fallback heuristic extraction keeps the project functional without a fine-tuned checkpoint
        fields = {}
        dni = self._extract_dni(text)
        nombre = self._extract_name(text)
        if dni:
            fields["dni"] = LayoutField("dni", dni, 0.76, source="heuristic_layoutlm")
        if nombre:
            fields["nombre"] = LayoutField("nombre", nombre, 0.67, source="heuristic_layoutlm")
        return fields

    def _extract_dni(self, text: str) -> str | None:
        m = re.search(r"\b\d{7,8}\b", text)
        return m.group(0) if m else None

    def _extract_name(self, text: str) -> str | None:
        # very lightweight fallback; replace with a fine-tuned token classification model
        m = re.search(r"(?:nombre|apellid[oa])[:\s]+([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑñ\- ]{2,})", text, re.IGNORECASE)
        return m.group(1).strip() if m else None
