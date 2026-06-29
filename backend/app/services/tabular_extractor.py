from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Optional
import math
import re

import numpy as np

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


@dataclass
class LayoutField:
    field_name: str
    value: Any
    confidence: float = 0.0
    source: str = "heuristic"
    page: Optional[int] = None
    bbox: Optional[list[float]] = None
    ambiguous: bool = False
    candidates: Optional[list[Any]] = None


@dataclass
class TextLine:
    text: str
    bbox: list[float]  # canonical vertical-first [y1, x1, y2, x2]
    page: int = 0
    conf: float = 0.0
    source: str = "ocr"

    @property
    def y1(self) -> float:
        return float(self.bbox[0])

    @property
    def x1(self) -> float:
        return float(self.bbox[1])

    @property
    def y2(self) -> float:
        return float(self.bbox[2])

    @property
    def x2(self) -> float:
        return float(self.bbox[3])

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)


class TabularExtractor:
    """
    Reconstrucción heurística y tolerante a fallos de filas de electores.
    Reglas clave:
    - No depende de tablas ni rejillas.
    - Detecta dinámicamente qué par de coordenadas representa el eje vertical.
    - Si no se encuentra checkbox, la fila igualmente se conserva con voto=False.
    - Nunca devuelve 0 filas mientras exista texto utilizable en la página.
    """

    _HEADER_TERMS = {
        "REGISTRO", "NACIONAL", "ELECTORES", "CAMARA", "CÁMARA",
        "ELECTORAL", "PADRON", "PADRÓN", "DISTRITO", "CIRCUITO",
        "MESA", "SECCION", "SECCIÓN", "DEFINITIVO", "INSCRIPTOS",
        "ELECCIONES", "PRIMARIAS", "PROVINCIA", "CUARTO", "EJEMPLAR",
        "APELLIDOS", "NOMBRES", "DOMICILIO", "VOTO", "VOTÓ",
    }

    _DOC_RE = re.compile(r"\b(?:DNI|LE|LC|CI|PASAPORTE)?\s*[:\-]?\s*([0-9\.]{6,12})\b", re.IGNORECASE)
    _ORDER_RE = re.compile(r"\b(?:N[°ºo]?|NRO\.?\s*ORDEN|ORDEN|N\s*ORDEN)\s*[:\-]?\s*(\d{1,6})\b", re.IGNORECASE)
    _YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
    _ONLY_NOISE_RE = re.compile(r"^[\W_]+$")

    def extract(
        self,
        pages: list[np.ndarray],
        tokens: list[dict[str, Any]],
        visual: dict[str, Any] | None,
        raw_text: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, LayoutField]]:
        visual = visual or {}
        header_fields = self._extract_header_fields(tokens, raw_text=raw_text)

        tokens_by_page: dict[int, list[dict[str, Any]]] = {}
        for token in tokens or []:
            page_idx = int(token.get("page", 0) or 0)
            tokens_by_page.setdefault(page_idx, []).append(token)

        raw_page_lines = visual.get("page_lines") or []
        checkbox_pool = self._normalize_checkboxes(visual)
        checkboxes_by_page: dict[int, list[dict[str, Any]]] = {}
        for cb in checkbox_pool:
            checkboxes_by_page.setdefault(int(cb.get("page", 0) or 0), []).append(cb)

        rows: list[dict[str, Any]] = []

        for page_idx, page in enumerate(pages or []):
            page_height = int(page.shape[0]) if hasattr(page, "shape") and len(page.shape) >= 2 else 0
            page_width = int(page.shape[1]) if hasattr(page, "shape") and len(page.shape) >= 2 else 0

            raw_lines = self._normalize_page_line_items(raw_page_lines[page_idx] if page_idx < len(raw_page_lines) else [])
            page_tokens = self._normalize_token_items(tokens_by_page.get(page_idx, []))
            page_checkboxes = checkboxes_by_page.get(page_idx, [])

            bbox_mode = self._infer_bbox_mode(raw_lines + page_tokens + page_checkboxes, page_width, page_height)

            lines = self._build_text_lines(raw_lines, bbox_mode, page_idx)
            if not lines:
                lines = self._group_tokens_into_lines(page_tokens, bbox_mode, page_idx)

            page_rows = self._extract_rows_from_lines(
                page_idx=page_idx,
                page=page,
                lines=lines,
                checkbox_pool=page_checkboxes,
            )

            if not page_rows:
                page_rows = self._fallback_rows_from_text_lines(
                    page_idx=page_idx,
                    page=page,
                    raw_lines=lines,
                    checkbox_pool=page_checkboxes,
                )

            if not page_rows:
                page_rows = self._fallback_rows_from_text(
                    page_idx=page_idx,
                    page=page,
                    raw_text=raw_text or "",
                    checkbox_pool=page_checkboxes,
                )

            rows.extend(page_rows)

        rows = self._deduplicate_rows(rows)

        # Seguridad absoluta: nunca devolver 0 filas si hay texto.
        if not rows and raw_text and self._normalize_spaces(raw_text):
            fallback = self._fallback_rows_from_text(
                page_idx=0,
                page=pages[0] if pages else np.zeros((1, 1, 3), dtype=np.uint8),
                raw_text=raw_text,
                checkbox_pool=checkbox_pool,
            )
            rows.extend(self._deduplicate_rows(fallback))

        return rows, header_fields

    def extract_tabular(
        self,
        pages: list[np.ndarray],
        tokens: list[dict[str, Any]],
        visual: dict[str, Any],
        raw_text: str | None = None,
    ):
        return self.extract(pages, tokens, visual, raw_text=raw_text)

    # ------------------------------------------------------------------
    # Header fields
    # ------------------------------------------------------------------
    def _extract_header_fields(
        self,
        tokens: list[dict[str, Any]],
        raw_text: str | None = None,
    ) -> dict[str, LayoutField]:
        text = self._normalize_spaces(raw_text or self._tokens_to_text(tokens))
        fields: dict[str, LayoutField] = {}

        distrito = self._find_pattern(text, r"DISTRITO\s*[:\-]?\s*([0-9A-ZÁÉÍÓÚÑ\- ]+)")
        seccion = self._find_pattern(text, r"SECCI[ÓO]N\s*[:\-]?\s*([0-9A-ZÁÉÍÓÚÑ\- ]+)")
        circuito = self._find_pattern(text, r"CIRCUITO\s*[:\-]?\s*([0-9A-ZÁÉÍÓÚÑ\- ]+)")
        mesa = self._find_pattern(text, r"MESA\s*[:\-]?\s*([0-9A-ZÁÉÍÓÚÑ\- ]+)")

        if distrito:
            fields["distrito"] = LayoutField("distrito", distrito, 0.95, source="header")
        if seccion:
            fields["seccion"] = LayoutField("seccion", seccion, 0.95, source="header")
        if circuito:
            fields["circuito"] = LayoutField("circuito", circuito, 0.95, source="header")
        if mesa:
            fields["mesa"] = LayoutField("mesa", mesa, 0.95, source="header")

        return fields

    def _find_pattern(self, text: str, pattern: str) -> str | None:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return None
        return self._normalize_spaces(m.group(1)).strip(" :-")

    # ------------------------------------------------------------------
    # Row extraction
    # ------------------------------------------------------------------
    def _extract_rows_from_lines(
        self,
        page_idx: int,
        page: np.ndarray,
        lines: list[TextLine],
        checkbox_pool: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not lines:
            return []

        bands = self._group_lines_by_band(lines)
        rows: list[dict[str, Any]] = []

        for band_index, band in enumerate(bands):
            row = self._parse_row(
                page_idx=page_idx,
                page=page,
                fragments=band,
                checkbox_pool=checkbox_pool,
                band_index=band_index,
                block_index=0,
                synthetic=False,
            )
            if row is not None:
                rows.append(row)

        return rows

    def _fallback_rows_from_text_lines(
        self,
        page_idx: int,
        page: np.ndarray,
        raw_lines: list[TextLine],
        checkbox_pool: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i, line in enumerate(raw_lines or []):
            txt = self._normalize_spaces(line.text)
            if not txt or self._is_header_like(txt):
                continue
            row = self._parse_row(
                page_idx=page_idx,
                page=page,
                fragments=[line],
                checkbox_pool=checkbox_pool,
                band_index=i,
                block_index=0,
                synthetic=True,
            )
            if row is not None:
                rows.append(row)
        return rows

    def _fallback_rows_from_text(
        self,
        page_idx: int,
        page: np.ndarray,
        raw_text: str,
        checkbox_pool: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        text = self._normalize_spaces(raw_text)
        if not text:
            return []

        lines = [self._normalize_spaces(line) for line in text.splitlines()]
        lines = [line for line in lines if line and not self._is_header_like(line)]
        
        if not lines:
            parts = [p for p in re.split(r"\s{2,}", text) if self._normalize_spaces(p)]
            lines = [self._normalize_spaces(p) for p in parts if not self._is_header_like(p)]

        if not lines:
            return []

        page_h = page.shape[0] if hasattr(page, "shape") and len(page.shape) >= 2 else max(1, len(lines))
        synthetic_step = max(14.0, float(page_h) / max(1, len(lines)))

        rows: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            y = float(i * synthetic_step + synthetic_step / 2.0)
            fragments = self._line_to_fragments(line, y=y)
            row = self._parse_row(
                page_idx=page_idx,
                page=page,
                fragments=fragments,
                checkbox_pool=checkbox_pool,
                band_index=i,
                block_index=0,
                synthetic=True,
            )
            if row is not None:
                rows.append(row)
        return rows

    def _parse_row(
        self,
        page_idx: int,
        page: np.ndarray,
        fragments: list[TextLine],
        checkbox_pool: list[dict[str, Any]],
        band_index: int,
        block_index: int,
        synthetic: bool = False,
    ) -> dict[str, Any] | None:
        fragments = [f for f in fragments if self._normalize_spaces(f.text)]
        if not fragments:
            return None

        fragments = sorted(fragments, key=lambda f: (f.x1, f.y1))
        text = self._normalize_spaces(" ".join(f.text for f in fragments))
        if not text or self._is_header_like(text) or self._ONLY_NOISE_RE.match(text):
            return None

        order = self._extract_order(text)
        dni = self._extract_dni(text)
        nombre = self._extract_name(text)
        domicilio = self._extract_domicilio(text)
        year = self._extract_year(text)

        y1 = min(f.y1 for f in fragments)
        y2 = max(f.y2 for f in fragments)
        x1 = min(f.x1 for f in fragments)
        x2 = max(f.x2 for f in fragments)
        row_bbox = [y1, x1, y2, x2]

        vote, matched_cb = self._match_checkbox_for_row(row_bbox, checkbox_pool)

        confidence = self._row_confidence(order, dni, nombre, text)
        row = {
            "page": page_idx,
            "band_index": band_index,
            "block_index": block_index,
            "texto_fila": text,
            "nro_orden": order,
            "documento_dni": dni, # Alineado con el nombre de campo que espera el backend
            "apellido_nombre": nombre,
            "domicilio": domicilio,
            "anio": year,
            "voto": bool(vote),
            "bbox": row_bbox,
            "bbox_mode": "canonical_yx",
            "confidence": confidence,
            "synthetic": synthetic,
            "matched_checkbox": matched_cb,
        }
        return row

    def _match_checkbox_for_row(self, row_bbox: list[float], checkbox_pool: list[dict[str, Any]]) -> tuple[bool, dict[str, Any] | None]:
        if not checkbox_pool:
            return False, None

        row_y1, _, row_y2, _ = row_bbox
        row_center = (row_y1 + row_y2) / 2.0
        best: tuple[float, dict[str, Any]] | None = None

        for cb in checkbox_pool:
            bbox = cb.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            cb_y1 = float(bbox[0])
            cb_y2 = float(bbox[2])
            cb_center = (cb_y1 + cb_y2) / 2.0
            dist = abs(cb_center - row_center)
            cb_h = max(1.0, cb_y2 - cb_y1)
            tolerance = max(20.0, cb_h * 1.5)

            if dist <= tolerance:
                if best is None or dist < best[0]:
                    best = (dist, cb)

        if best is None:
            return False, None
        return bool(best[1].get("checked", False)), best[1]

    # ------------------------------------------------------------------
    # BBoxes / axis inference
    # ------------------------------------------------------------------
    def _normalize_checkboxes(self, visual: dict[str, Any]) -> list[dict[str, Any]]:
        raw = visual.get("campos_marcados")
        if isinstance(raw, dict) and raw.get("field_name") == "campos_marcados":
            raw = raw.get("value")
        if isinstance(raw, dict) and "value" in raw:
            raw = raw["value"]
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raw = [raw]

        out: list[dict[str, Any]] = []
        for item in raw:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "__dict__") and not isinstance(item, dict):
                item = dict(item.__dict__)
            if not isinstance(item, dict):
                continue

            bbox = self._normalize_raw_bbox(item.get("bbox") or item.get("box") or item.get("polygon"))
            if bbox is None:
                continue

            out.append({
                "kind": item.get("kind", "checkbox"),
                "bbox": bbox,
                "checked": bool(item.get("checked", item.get("value", False))),
                "page": int(item.get("page", 0) or 0),
                "score": float(item.get("score", item.get("confidence", 0.0)) or 0.0),
            })
        return out

    def _infer_bbox_mode(self, items: list[dict[str, Any]], page_width: int, page_height: int) -> str:
        if not items:
            return "standard"

        best_mode = "standard"
        best_score = float("-inf")

        for mode in ("standard", "rotated"):
            fragments = []
            for item in items:
                bbox = self._normalize_raw_bbox(item.get("bbox"))
                if bbox is None:
                    continue
                canon = self._canonicalize_bbox(bbox, mode)
                if canon is None:
                    continue
                text = str(item.get("text") or item.get("value") or "").strip()
                fragments.append(TextLine(
                    text=text or "*", 
                    bbox=canon, 
                    page=int(item.get("page", 0) or 0), 
                    conf=float(item.get("conf", item.get("score", 0.0)) or 0.0)
                ))

            if not fragments:
                continue

            clusters = self._cluster_by_vertical_position(fragments)
            multi = sum(1 for c in clusters if len(c) >= 2)
            avg_size = mean([len(c) for c in clusters]) if clusters else 0.0
            spread = self._vertical_spread([f.cy for f in fragments])
            compactness = self._mean_within_cluster_spread(clusters)
            singleton_penalty = sum(1 for c in clusters if len(c) == 1)

            score = (multi * 120.0) + (avg_size * 18.0) + (spread * 0.08) - (singleton_penalty * 8.0) - (compactness * 0.5)

            if page_width > 0 and page_height > 0:
                in_bounds = sum(1 for f in fragments if -5 <= f.y1 <= page_height + 5 and -5 <= f.x1 <= page_width + 5)
                score += in_bounds * 0.5

            if score > best_score:
                best_score = score
                best_mode = mode

        return best_mode

    def _normalize_raw_bbox(self, bbox: Any) -> list[float] | None:
        if bbox is None:
            return None
        if hasattr(bbox, "model_dump"):
            bbox = bbox.model_dump()
        if isinstance(bbox, dict):
            if {"x1", "y1", "x2", "y2"}.issubset(bbox.keys()):
                bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
            elif {"left", "top", "right", "bottom"}.issubset(bbox.keys()):
                bbox = [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]]
            elif {"x", "y", "w", "h"}.issubset(bbox.keys()):
                bbox = [bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]]

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            return [float(v) for v in bbox]
        except Exception:
            return None

    def _canonicalize_bbox(self, bbox: list[float], mode: str) -> list[float] | None:
        if len(bbox) != 4:
            return None
        a, b, c, d = [float(v) for v in bbox]
        if mode == "standard":
            return [a, b, c, d]
        return [a, b, c, d] if a > b else [b, a, d, c] # Corrección de rotación

    def _normalize_page_line_items(self, page_lines: Any) -> list[dict[str, Any]]:
        if page_lines is None:
            return []
        if isinstance(page_lines, dict) and "lines" in page_lines:
            page_lines = page_lines["lines"]
        if not isinstance(page_lines, list):
            return []

        out: list[dict[str, Any]] = []
        for item in page_lines:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "__dict__") and not isinstance(item, dict):
                item = dict(item.__dict__)
            if not isinstance(item, dict):
                continue

            text = self._normalize_spaces(item.get("text") or item.get("line") or item.get("value") or "")
            bbox = self._normalize_raw_bbox(item.get("bbox") or item.get("box") or item.get("polygon"))
            if not text or bbox is None:
                continue

            out.append({
                "text": text,
                "bbox": bbox,
                "page": int(item.get("page", 0) or 0),
                "conf": float(item.get("conf", item.get("confidence", 0.0)) or 0.0),
                "source": str(item.get("source", "line")),
            })
        return out

    def _normalize_token_items(self, tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in tokens or []:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "dict"):
                item = item.dict()
            elif hasattr(item, "__dict__") and not isinstance(item, dict):
                item = dict(item.__dict__)
            if not isinstance(item, dict):
                continue
            text = self._normalize_spaces(item.get("text") or item.get("value") or item.get("content") or "")
            bbox = self._normalize_raw_bbox(item.get("bbox") or item.get("box") or item.get("polygon"))
            if not text or bbox is None:
                continue
            out.append({
                "text": text,
                "bbox": bbox,
                "page": int(item.get("page", 0) or 0),
                "conf": float(item.get("conf", item.get("confidence", 0.0)) or 0.0),
                "source": str(item.get("source", "ocr")),
            })
        return out

    def _build_text_lines(self, items: list[dict[str, Any]], bbox_mode: str, page_idx: int) -> list[TextLine]:
        lines: list[TextLine] = []
        for item in items or []:
            bbox = self._canonicalize_bbox(self._normalize_raw_bbox(item.get("bbox")) or [], bbox_mode)
            if bbox is None:
                continue
            lines.append(TextLine(
                text=self._normalize_spaces(item.get("text") or ""),
                bbox=bbox,
                page=int(item.get("page", page_idx) or page_idx),
                conf=float(item.get("conf", 0.0) or 0.0),
                source=str(item.get("source", "line")),
            ))
        return [line for line in lines if self._normalize_spaces(line.text)]

    def _group_tokens_into_lines(self, items: list[dict[str, Any]], bbox_mode: str, page_idx: int) -> list[TextLine]:
        fragments = self._build_text_lines(items, bbox_mode, page_idx)
        if not fragments:
            return []
        bands = self._cluster_by_vertical_position(fragments)
        out: list[TextLine] = []
        for band in bands:
            band = sorted(band, key=lambda f: (f.x1, f.y1))
            text = self._normalize_spaces(" ".join(f.text for f in band))
            y1 = min(f.y1 for f in band)
            x1 = min(f.x1 for f in band)
            y2 = max(f.y2 for f in band)
            x2 = max(f.x2 for f in band)
            out.append(TextLine(
                text=text, bbox=[y1, x1, y2, x2], page=page_idx, 
                conf=mean([f.conf for f in band]) if band else 0.0, source="grouped"
            ))
        return [line for line in out if self._normalize_spaces(line.text)]

    def _cluster_by_vertical_position(self, fragments: list[TextLine], tolerance: float | None = None) -> list[list[TextLine]]:
        if not fragments:
            return []
        ordered = sorted(fragments, key=lambda f: (f.cy, f.x1))
        clusters: list[list[TextLine]] = []
        current: list[TextLine] = [ordered[0]]
        current_y = ordered[0].cy
        current_tol = tolerance if tolerance is not None else max(15.0, ordered[0].height * 0.75)

        for frag in ordered[1:]:
            frag_y = frag.cy
            frag_tol = tolerance if tolerance is not None else max(15.0, frag.height * 0.75)
            if abs(frag_y - current_y) <= current_tol:
                current.append(frag)
                current_y = sum(f.cy for f in current) / len(current)
                current_tol = max(current_tol, frag_tol)
            else:
                clusters.append(current)
                current = [frag]
                current_y = frag_y
                current_tol = frag_tol

        if current:
            clusters.append(current)
        return clusters

    def _group_lines_by_band(self, fragments: list[TextLine]) -> list[list[TextLine]]:
        return self._cluster_by_vertical_position(fragments, tolerance=18.0)

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------
    def _extract_order(self, text: str) -> str:
        m = self._ORDER_RE.search(text)
        if m:
            return m.group(1)
        m = re.match(r"^\s*(\d{1,6})\b", text)
        return m.group(1) if m else ""

    def _extract_dni(self, text: str) -> str:
        m = self._DOC_RE.search(text)
        return re.sub(r"\D", "", m.group(1)) if m else ""

    def _extract_name(self, text: str) -> str:
        clean = self._normalize_spaces(text)
        # Limpiar el DNI
        clean = re.sub(r"\b(?:DNI|LE|LC|CI|PASAPORTE)?\s*[:\-]?\s*[0-9\.]{6,12}\b", "", clean, flags=re.IGNORECASE)
        # Limpiar el Orden
        clean = re.sub(r"\b(?:N[°ºo]?|NRO\.?\s*ORDEN|ORDEN|N\s*ORDEN)\s*[:\-]?\s*\d{1,6}\b", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"^\s*\d{1,6}\b", "", clean)
        
        clean = self._normalize_spaces(clean)
        clean = re.sub(r"^[\s,;:\-]+", "", clean).strip()
        
        if len(clean) < 3:
            return ""
        return clean

    def _extract_domicilio(self, text: str) -> str:
        upper = text.upper()
        for marker in ("DOMICILIO", "CALLE", "AV ", "AV.", "BARRIO", "PISO", "DPTO", "DTO", "KM"):
            idx = upper.find(marker)
            if idx >= 0:
                return self._normalize_spaces(text[idx:])
        return ""

    def _extract_year(self, text: str) -> str:
        m = self._YEAR_RE.search(text)
        return m.group(1) if m else ""

    def _row_confidence(self, order: str, dni: str, nombre: str, text: str) -> float:
        score = 0.20
        if order: score += 0.25
        if dni: score += 0.35
        if nombre: score += 0.15
        if len(text) > 12: score += 0.05
        return min(0.99, score)

    # ------------------------------------------------------------------
    # Normalization / helpers
    # ------------------------------------------------------------------
    def _line_to_fragments(self, line: str, y: float) -> list[TextLine]:
        parts = [p for p in self._normalize_spaces(line).split(" ") if p]
        fragments: list[TextLine] = []
        cursor_x = 0.0
        for part in parts:
            width = max(8.0, len(part) * 7.0)
            fragments.append(TextLine(
                text=part,
                bbox=[y - 5.0, cursor_x, y + 5.0, cursor_x + width],
                conf=0.0,
                source="synthetic",
            ))
            cursor_x += width + 6.0
        return fragments

    def _tokens_to_text(self, tokens: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for token in tokens or []:
            t = str(token.get("text") or token.get("value") or token.get("content") or "").strip()
            if t:
                parts.append(t)
        return self._normalize_spaces(" ".join(parts))

    def _deduplicate_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []

        seen: set[tuple[int, str, str, str]] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            key = (
                int(row.get("page", 0) or 0),
                self._normalize_spaces(str(row.get("nro_orden", ""))),
                self._normalize_spaces(str(row.get("documento_dni", ""))),
                self._normalize_spaces(str(row.get("texto_fila", ""))),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _is_header_like(self, text: str) -> bool:
        u = text.upper()
        hits = sum(1 for term in self._HEADER_TERMS if term in u)
        return hits >= 2 or (hits >= 1 and len(u) < 120)

    def _vertical_spread(self, values: list[float]) -> float:
        if not values:
            return 0.0
        arr = np.array(values, dtype=float)
        return float(np.percentile(arr, 90) - np.percentile(arr, 10))

    def _mean_within_cluster_spread(self, clusters: list[list[TextLine]]) -> float:
        if not clusters:
            return 0.0
        spreads: list[float] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            ys = [f.cy for f in cluster]
            spreads.append(max(ys) - min(ys))
        return float(mean(spreads)) if spreads else 0.0