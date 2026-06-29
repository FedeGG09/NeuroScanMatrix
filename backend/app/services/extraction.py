from __future__ import annotations
from typing import Any
import re
from app.services.layoutlm_extractor import LayoutLMv3Extractor, LayoutField

class SemanticExtractor:
    def __init__(self, layout_extractor: LayoutLMv3Extractor):
        self.layout_extractor = layout_extractor

    def extract(self, tokens: list[dict[str, Any]], visual: dict[str, Any], doc_type: str) -> dict[str, LayoutField]:
        fields = self.layout_extractor.extract_from_ocr(tokens)
        full_text = " ".join(t.get("text", "") for t in tokens)
        # augment with document-specific heuristics
        if "firma" in full_text.lower():
            fields["firma_presente"] = LayoutField("firma_presente", True, 0.72, source="heuristic")
        if visual.get("checked_options"):
            fields["campos_marcados"] = LayoutField(
                "campos_marcados",
                visual["checked_options"],
                0.78,
                source="visual",
                ambiguous=False,
            )
        observations = self._extract_observations(full_text)
        if observations:
            fields["observaciones"] = LayoutField("observaciones", observations, 0.62, source="heuristic")
        return fields

    def _extract_observations(self, text: str) -> str | None:
        patterns = [
            r"(?:observaciones?|notas?)[:\s]+(.+)",
            r"(?:comentarios?)[:\s]+(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                val = m.group(1).strip()
                return val[:500]
        return None
