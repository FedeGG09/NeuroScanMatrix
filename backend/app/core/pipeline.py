from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from statistics import pstdev
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output

from app.core.config import settings
from app.services.document_classifier import DocumentClassifier
from app.services.extraction import SemanticExtractor
from app.services.layoutlm_extractor import LayoutLMv3Extractor
from app.services.preprocessing import preprocess_page
from app.services.tabular_extractor import TabularExtractor
from app.services.validation import field_to_dict, validate_fields
from app.utils.pdf_utils import is_pdf, pdf_to_images


# ---------------------------------------------------------------------
# Blindaje global de Tesseract
# ---------------------------------------------------------------------
DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_TESSDATA_PREFIX = r"C:\Program Files\Tesseract-OCR\tessdata"

_tesseract_cmd = getattr(settings, "tesseract_cmd", None) or os.getenv("TESSERACT_CMD") or DEFAULT_TESSERACT_CMD
pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

if os.path.isdir(DEFAULT_TESSDATA_PREFIX):
    os.environ.setdefault("TESSDATA_PREFIX", DEFAULT_TESSDATA_PREFIX)

print(f"[PIPELINE] Tesseract cmd configurado: {pytesseract.pytesseract.tesseract_cmd}", flush=True)
print(f"[PIPELINE] TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX', '(no definido)')}", flush=True)


_PIPELINE: "DocumentPipeline | None" = None


class DocumentPipeline:
    """
    Pipeline lineal y explícito para padrones.
    No hay recursión entre process() y process_document().
    """

    def __init__(self) -> None:
        self.tabular_extractor = TabularExtractor()

        # Estos componentes se conservan para compatibilidad con tu stack actual.
        self.doc_classifier = DocumentClassifier(
            settings.layout_classifier_model,
            device=getattr(settings, "device", "cpu"),
        )
        self.layout_extractor = LayoutLMv3Extractor(
            settings.layoutlmv3_model,
            device=getattr(settings, "device", "cpu"),
        )
        self.semantic_extractor = SemanticExtractor(self.layout_extractor)

        self.ocr_lang = getattr(settings, "tesseract_lang", "spa+eng")
        self.ocr_timeout = int(getattr(settings, "tesseract_timeout", 60))
        self.review_threshold = float(getattr(settings, "review_threshold", 0.75))
        self.max_pages = int(getattr(settings, "max_pages", 20))

    # -----------------------------------------------------------------
    # Punto de entrada interno de la clase
    # -----------------------------------------------------------------
    def run(self, file_path: str, document_id: str) -> dict[str, Any]:
        print(f"[PIPELINE] Iniciando run() document_id={document_id} file_path={file_path}", flush=True)

        pages = self._load_pages(file_path)
        print(f"[PIPELINE] Páginas cargadas: {len(pages)}", flush=True)

        all_tokens: list[dict[str, Any]] = []
        raw_text_parts: list[str] = []
        visual: dict[str, Any] = {
            "pages": [],
            "page_lines": [],
            "checkboxes": [],
            "checked_options": [],
            "campos_marcados": [],
            "page_shapes": [],
        }

        doc_type = "generic_form"
        doc_conf = 0.0

        for page_idx, page in enumerate(pages):
            print(f"\n[PIPELINE] --- Página {page_idx + 1}/{len(pages)} ---", flush=True)
            self._log_image_state(page, prefix="[PIPELINE] Imagen original")

            try:
                preprocessed = preprocess_page(page)
            except Exception:
                print("[PIPELINE] Error en preprocess_page()", flush=True)
                print(traceback.format_exc(), flush=True)
                raise

            sanitized = self._sanitize_for_ocr(preprocessed)
            self._log_image_state(sanitized, prefix="[PIPELINE] Imagen sanitizada")
            pil_rgb = self._to_pil_rgb(sanitized)
            pil_gray = ImageOps.grayscale(pil_rgb)

            if page_idx == 0:
                try:
                    doc_pred = self.doc_classifier.predict(sanitized, text_hint="")
                    doc_type = getattr(doc_pred, "document_type", "generic_form") or "generic_form"
                    doc_conf = float(getattr(doc_pred, "confidence", 0.0) or 0.0)
                    print(f"[PIPELINE] Clasificación documento: {doc_type} (conf={doc_conf:.3f})", flush=True)
                except Exception:
                    print("[PIPELINE] Falló document classifier, continúo con generic_form", flush=True)
                    print(traceback.format_exc(), flush=True)
                    doc_type = "generic_form"
                    doc_conf = 0.0

            tokens, page_lines, page_text = self._ocr_page_with_diagnostics(
                pil_gray=pil_gray,
                pil_rgb=pil_rgb,
                page_idx=page_idx,
            )

            all_tokens.extend(tokens)
            raw_text_parts.append(page_text)

            visual["page_lines"].append(page_lines)

            try:
                page_visual = self._detect_page_visuals(sanitized, page_idx)
            except Exception:
                print("[PIPELINE] Falló detección visual de checkboxes, continúo sin bloquear texto", flush=True)
                print(traceback.format_exc(), flush=True)
                page_visual = {"checkboxes": [], "checked_options": []}

            visual["pages"].append(page_visual)
            visual["checkboxes"].extend(page_visual.get("checkboxes", []))
            visual["checked_options"].extend(page_visual.get("checked_options", []))
            visual["page_shapes"].append(
                {
                    "page": page_idx,
                    "height": int(sanitized.shape[0]),
                    "width": int(sanitized.shape[1]),
                    "channels": int(sanitized.shape[2]) if sanitized.ndim == 3 else 1,
                    "dtype": str(sanitized.dtype),
                }
            )

        raw_text = "\n".join([t for t in raw_text_parts if t]).strip()
        print(f"\n[PIPELINE] OCR global finalizado. Largo raw_text={len(raw_text)}", flush=True)
        print(f"[PIPELINE] Tokens acumulados={len(all_tokens)}", flush=True)

        sem_fields = self._safe_semantic_extract(all_tokens, visual, doc_type)

        campos_marcados = self._extract_campos_marcados(sem_fields)
        if not campos_marcados:
            campos_marcados = list(visual.get("checked_options") or visual.get("checkboxes") or [])
        visual["campos_marcados"] = campos_marcados

        print(f"[PIPELINE] campos_marcados detectados={len(campos_marcados)}", flush=True)

        try:
            rows, header_fields = self.tabular_extractor.extract(
                pages=pages,
                tokens=all_tokens,
                visual=visual,
                raw_text=raw_text,
            )
        except Exception:
            print("[PIPELINE] Error en TabularExtractor.extract()", flush=True)
            print(traceback.format_exc(), flush=True)
            rows = []
            header_fields = {}

        if not rows:
            print("[PIPELINE] TabularExtractor devolvió 0 filas. Activando fallback permisivo.", flush=True)
            rows = self._fallback_rows_from_visual(visual=visual, raw_text=raw_text)

        if not rows:
            # Esto ya no debería ocurrir silenciosamente.
            raise RuntimeError(
                "OCR completado pero no se pudieron reconstruir filas. "
                "Revisá el preprocesado, el OCR o el extractor tabular."
            )

        try:
            global_score, needs_review = validate_fields(header_fields, threshold=self.review_threshold)
        except Exception:
            print("[PIPELINE] validate_fields() falló, continuo con defaults", flush=True)
            print(traceback.format_exc(), flush=True)
            global_score = 0.0
            needs_review = True

        field_payload = {}
        if isinstance(header_fields, dict):
            field_payload = {k: field_to_dict(v) for k, v in header_fields.items()}

        checked_count = sum(1 for r in rows if r.get("voto") is True)

        result: dict[str, Any] = {
            "document_id": document_id,
            "document_type": doc_type,
            "document_type_confidence": doc_conf,
            "confidence_global": global_score,
            "needs_review": needs_review,
            "fields": field_payload,
            "raw_text": raw_text,
            "visual_elements": visual,
            "pages_processed": len(pages),
            "status": "ok",
            "rows": rows,
            "metadata": {
                "rows_extracted": len(rows),
                "checked_count": checked_count,
                "unchecked_count": max(0, len(rows) - checked_count),
            },
        }

        print(
            f"[PIPELINE] Finalizado OK: rows={len(rows)} checked={checked_count} needs_review={needs_review}",
            flush=True,
        )
        return result

    # Compatibilidad con código viejo que llame process()
    def process(self, file_path: str, document_id: str) -> dict[str, Any]:
        return self.run(file_path, document_id)

    # -----------------------------------------------------------------
    # Carga y sanitización de páginas
    # -----------------------------------------------------------------
    def _load_pages(self, file_path: str) -> list[np.ndarray]:
        if is_pdf(file_path):
            pages = pdf_to_images(file_path, max_pages=self.max_pages)
            if not pages:
                raise RuntimeError(f"El PDF no devolvió páginas: {file_path}")
            return [self._sanitize_for_ocr(page) for page in pages]

        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"No se pudo leer el archivo: {file_path}")
        return [self._sanitize_for_ocr(img)]

    def _sanitize_for_ocr(self, image: Any) -> np.ndarray:
        """
        Convierte cualquier entrada razonable a un ndarray uint8 válido para OCR.
        - elimina alpha
        - evita float32
        - asegura 2D o 3 canales
        """
        if isinstance(image, Image.Image):
            arr = np.array(image)
        elif isinstance(image, np.ndarray):
            arr = image
        else:
            raise TypeError(f"Tipo de imagen no soportado: {type(image)!r}")

        if arr.size == 0:
            raise ValueError("La imagen llegó vacía al pipeline.")

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            return arr

        if arr.ndim == 3:
            channels = arr.shape[2]
            if channels == 4:
                # Quitamos alpha; el orden RGB/BGR no es crítico para Tesseract tras convertir a gray.
                arr = arr[:, :, :3]
                channels = 3
            if channels == 3:
                return arr
            raise ValueError(f"Cantidad de canales no soportada: {channels}")

        raise ValueError(f"Dimensiones de imagen no soportadas: shape={arr.shape}")

    def _to_pil_rgb(self, image: np.ndarray) -> Image.Image:
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(image).convert("RGB")

    def _log_image_state(self, image: Any, prefix: str = "[PIPELINE]") -> None:
        if isinstance(image, np.ndarray):
            info = {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "ndim": image.ndim,
                "min": int(np.min(image)) if image.size else None,
                "max": int(np.max(image)) if image.size else None,
            }
            if image.ndim == 3:
                info["channels"] = int(image.shape[2])
            else:
                info["channels"] = 1
            print(f"{prefix} {info}", flush=True)
        elif isinstance(image, Image.Image):
            print(
                f"{prefix} PIL mode={image.mode} size={image.size}",
                flush=True,
            )
        else:
            print(f"{prefix} tipo={type(image)!r}", flush=True)

    # -----------------------------------------------------------------
    # OCR robusto con diagnóstico profundo
    # -----------------------------------------------------------------
    def _ocr_page_with_diagnostics(
        self,
        pil_gray: Image.Image,
        pil_rgb: Image.Image,
        page_idx: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        """
        Ejecuta Tesseract de forma explícita y ruidosa.
        Si falla, imprime traceback completo y revienta.
        """
        attempts = [
            ("GRAY", pil_gray, 11),
            ("GRAY", pil_gray, 6),
            ("RGB", pil_rgb, 11),
            ("RGB", pil_rgb, 6),
        ]

        last_problem: Exception | None = None

        for mode_name, pil_img, psm in attempts:
            config = f"--oem 3 --psm {psm}"
            print(
                f"[PIPELINE] Llamando a Tesseract page={page_idx} mode={mode_name} psm={psm} lang={self.ocr_lang}",
                flush=True,
            )

            try:
                data = pytesseract.image_to_data(
                    pil_img,
                    lang=self.ocr_lang,
                    config=config,
                    output_type=Output.DICT,
                    timeout=self.ocr_timeout,
                )

                tokens = self._tesseract_dict_to_tokens(data, page_idx)
                lines = self._group_tokens_into_lines(tokens)
                raw_text = "\n".join([line["text"] for line in lines if line.get("text")]).strip()

                print(
                    f"[PIPELINE] Tesseract OK page={page_idx} mode={mode_name} psm={psm} "
                    f"tokens={len(tokens)} lines={len(lines)} raw_text_len={len(raw_text)}",
                    flush=True,
                )

                if raw_text:
                    sample = raw_text[:250].replace("\n", " | ")
                    print(f"[PIPELINE] raw_text sample: {sample}", flush=True)

                # Si esta combinación devolvió algo útil, la usamos.
                if tokens or raw_text:
                    return tokens, lines, raw_text

                print(
                    f"[PIPELINE] OCR vacío con mode={mode_name} psm={psm}; probando fallback...",
                    flush=True,
                )

            except Exception as exc:
                last_problem = exc
                print(
                    f"[PIPELINE] Error OCR page={page_idx} mode={mode_name} psm={psm}",
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)

        if last_problem is not None:
            raise RuntimeError(
                f"Tesseract falló en todas las variantes para la página {page_idx + 1}"
            ) from last_problem

        raise RuntimeError(f"Tesseract devolvió salida vacía para la página {page_idx + 1}")

    def _tesseract_dict_to_tokens(self, data: dict[str, list[Any]], page_idx: int) -> list[dict[str, Any]]:
        tokens: list[dict[str, Any]] = []
        if not data:
            return tokens

        n = len(data.get("text", []))
        for i in range(n):
            text = str(data.get("text", [""])[i]).strip()
            conf_raw = data.get("conf", ["-1"])[i]
            try:
                conf = float(conf_raw)
            except Exception:
                conf = -1.0

            if not text:
                continue
            if conf < 0:
                # Tesseract usa -1 para niveles no textuales; los filtramos.
                continue

            left = int(data.get("left", [0])[i])
            top = int(data.get("top", [0])[i])
            width = int(data.get("width", [0])[i])
            height = int(data.get("height", [0])[i])

            token = {
                "page": page_idx,
                "text": text,
                "conf": conf,
                "bbox": [left, top, left + width, top + height],
                "block_num": int(data.get("block_num", [0])[i]),
                "par_num": int(data.get("par_num", [0])[i]),
                "line_num": int(data.get("line_num", [0])[i]),
                "word_num": int(data.get("word_num", [0])[i]),
            }
            tokens.append(token)

        return tokens

    def _group_tokens_into_lines(self, tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Reconstruye líneas lógicas a partir de tokens de Tesseract.
        """
        if not tokens:
            return []

        groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for token in tokens:
            key = (
                int(token.get("block_num", 0)),
                int(token.get("par_num", 0)),
                int(token.get("line_num", 0)),
            )
            groups.setdefault(key, []).append(token)

        lines: list[dict[str, Any]] = []
        for key, group in sorted(groups.items(), key=lambda kv: kv[0]):
            group_sorted = sorted(group, key=lambda t: int(t["bbox"][0]))
            text = " ".join(t["text"] for t in group_sorted if t.get("text")).strip()
            if not text:
                continue

            x1 = min(int(t["bbox"][0]) for t in group_sorted)
            y1 = min(int(t["bbox"][1]) for t in group_sorted)
            x2 = max(int(t["bbox"][2]) for t in group_sorted)
            y2 = max(int(t["bbox"][3]) for t in group_sorted)

            lines.append(
                {
                    "page": group_sorted[0].get("page", 0),
                    "text": text,
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(np.mean([float(t.get("conf", 0.0)) for t in group_sorted])) if group_sorted else 0.0,
                    "source": "tesseract_line",
                    "key": key,
                }
            )

        return lines

    # -----------------------------------------------------------------
    # Semántica y visual
    # -----------------------------------------------------------------
    def _safe_semantic_extract(
        self,
        all_tokens: list[dict[str, Any]],
        visual: dict[str, Any],
        doc_type: str,
    ) -> dict[str, Any]:
        try:
            extracted = self.semantic_extractor.extract(all_tokens, visual, doc_type)
            return extracted if isinstance(extracted, dict) else {}
        except Exception:
            print("[PIPELINE] semantic_extractor.extract() falló", flush=True)
            print(traceback.format_exc(), flush=True)
            return {}

    def _extract_campos_marcados(self, sem_fields: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(sem_fields, dict):
            return []

        node = self._find_campos_marcados_node(sem_fields)
        if node is None:
            return []

        if hasattr(node, "model_dump"):
            node = node.model_dump()

        if isinstance(node, dict):
            if node.get("field_name") == "campos_marcados":
                node = node.get("value", [])
            elif "value" in node:
                node = node["value"]

        if not isinstance(node, list):
            node = [node]

        normalized: list[dict[str, Any]] = []
        for item in node:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif hasattr(item, "__dict__") and not isinstance(item, dict):
                item = dict(item.__dict__)

            if not isinstance(item, dict):
                continue

            bbox = item.get("bbox") or item.get("box") or item.get("polygon")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            try:
                bbox_f = [float(v) for v in bbox]
            except Exception:
                continue

            normalized.append(
                {
                    "kind": item.get("kind", "checkbox"),
                    "bbox": bbox_f,
                    "checked": bool(item.get("checked", item.get("value", False))),
                    "page": int(item.get("page", 0) or 0),
                    "score": float(item.get("score", item.get("confidence", 0.0)) or 0.0),
                }
            )

        return normalized

    def _find_campos_marcados_node(self, node: Any) -> Any:
        if node is None:
            return None

        if hasattr(node, "model_dump"):
            node = node.model_dump()
        elif hasattr(node, "__dict__") and not isinstance(node, dict):
            node = dict(node.__dict__)

        if isinstance(node, dict):
            if node.get("field_name") == "campos_marcados":
                return node.get("value", node)

            if "campos_marcados" in node:
                return node["campos_marcados"]

            fields = node.get("fields")
            if isinstance(fields, dict):
                found = self._find_campos_marcados_node(fields)
                if found is not None:
                    return found

            for value in node.values():
                found = self._find_campos_marcados_node(value)
                if found is not None:
                    return found

        elif isinstance(node, list):
            for item in node:
                found = self._find_campos_marcados_node(item)
                if found is not None:
                    return found

        return None

    def _detect_page_visuals(self, image: np.ndarray, page_idx: int) -> dict[str, Any]:
        """
        Detección visual de checkbox.
        Si falla, debe imprimirse el traceback fuera de esta función.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        contours, _ = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]

        candidates: list[dict[str, Any]] = []
        checked: list[dict[str, Any]] = []

        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)

            if not (8 <= ww <= 40 and 8 <= hh <= 40):
                continue

            ratio = ww / float(max(1, hh))
            if not (0.70 <= ratio <= 1.30):
                continue

            area = float(cv2.contourArea(cnt))
            box_area = float(max(1, ww * hh))
            fill_ratio = area / box_area
            if fill_ratio < 0.02:
                continue

            if x < w * 0.15 or x > w * 0.99:
                continue
            if y < h * 0.05 or y > h * 0.99:
                continue

            bbox = [float(x), float(y), float(x + ww), float(y + hh)]
            cand = {
                "kind": "checkbox",
                "bbox": bbox,
                "score": round(min(0.99, 0.30 + fill_ratio), 3),
                "page": page_idx,
            }
            candidates.append(cand)

            if self._is_checked_square(image, bbox):
                checked.append({**cand, "checked": True})

        return {
            "checkboxes": self._deduplicate_boxes(candidates),
            "checked_options": self._deduplicate_boxes(checked),
        }

    def _is_checked_square(self, image: np.ndarray, bbox: list[float]) -> bool:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False

        crop = image[
            max(0, y1 - 2):min(h, y2 + 2),
            max(0, x1 - 2):min(w, x2 + 2),
        ]
        if crop.size == 0:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        hh, ww = inv.shape[:2]
        inner = inv[
            max(1, int(hh * 0.18)):max(2, int(hh * 0.82)),
            max(1, int(ww * 0.18)):max(2, int(ww * 0.82)),
        ]
        if inner.size == 0:
            inner = inv

        dark_ratio = float(np.mean(inner > 0))
        if dark_ratio >= 0.06:
            return True

        lines = cv2.HoughLinesP(
            inner,
            1,
            np.pi / 180,
            threshold=8,
            minLineLength=max(3, int(min(inner.shape[:2]) * 0.35)),
            maxLineGap=2,
        )
        return lines is not None and len(lines) >= 1

    def _deduplicate_boxes(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items:
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            if any(self._boxes_overlap(bbox, other.get("bbox", [])) for other in out):
                continue
            out.append(item)
        return out

    def _boxes_overlap(self, a: list[float], b: list[float]) -> bool:
        if len(a) != 4 or len(b) != 4:
            return False

        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return False

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter_area / min(area_a, area_b) > 0.5

    # -----------------------------------------------------------------
    # Fallback permisivo para evitar 0 filas
    # -----------------------------------------------------------------
    def _fallback_rows_from_visual(self, visual: dict[str, Any], raw_text: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        page_lines = visual.get("page_lines") or []
        campos_marcados = visual.get("campos_marcados") or []

        for page_idx, lines in enumerate(page_lines):
            if not isinstance(lines, list):
                continue

            for line in lines:
                if not isinstance(line, dict):
                    continue

                text = str(line.get("text", "")).strip()
                if not text:
                    continue

                parsed = self._parse_row_text(text)
                rows.append(
                    {
                        "page": page_idx,
                        "nro_orden": parsed.get("nro_orden"),
                        "documento_dni": parsed.get("documento_dni"),
                        "apellido_nombre": parsed.get("apellido_nombre"),
                        "texto_fila": text,
                        "voto": self._fallback_voto_for_text_line(line, campos_marcados),
                        "source": "fallback_visual",
                    }
                )

        if rows:
            print(f"[PIPELINE] Fallback generó {len(rows)} filas desde líneas OCR", flush=True)
            return rows

        for line in (raw_text or "").splitlines():
            txt = line.strip()
            if not txt:
                continue
            parsed = self._parse_row_text(txt)
            rows.append(
                {
                    "page": 0,
                    "nro_orden": parsed.get("nro_orden"),
                    "documento_dni": parsed.get("documento_dni"),
                    "apellido_nombre": parsed.get("apellido_nombre"),
                    "texto_fila": txt,
                    "voto": False,
                    "source": "fallback_raw_text",
                }
            )

        print(f"[PIPELINE] Fallback raw_text generó {len(rows)} filas", flush=True)
        return rows

    def _fallback_voto_for_text_line(self, line: dict[str, Any], campos_marcados: list[dict[str, Any]]) -> bool:
        """
        Fallback suave: si no se puede asociar checkbox, devuelve False.
        """
        try:
            page = int(line.get("page", 0) or 0)
            bbox = line.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return False

            line_center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
            candidates = []
            for cb in campos_marcados:
                if int(cb.get("page", page) or 0) != page:
                    continue
                cb_bbox = cb.get("bbox")
                if not isinstance(cb_bbox, (list, tuple)) or len(cb_bbox) != 4:
                    continue
                cb_center_y = (float(cb_bbox[1]) + float(cb_bbox[3])) / 2.0
                distance = abs(cb_center_y - line_center_y)
                if distance <= 20.0:
                    candidates.append((distance, cb))

            if not candidates:
                return False

            candidates.sort(key=lambda x: x[0])
            return bool(candidates[0][1].get("checked", False))
        except Exception:
            print("[PIPELINE] _fallback_voto_for_text_line() falló", flush=True)
            print(traceback.format_exc(), flush=True)
            return False

    def _parse_row_text(self, text: str) -> dict[str, Any]:
        import re

        cleaned = re.sub(r"\s+", " ", (text or "")).strip()

        nro_orden = None
        documento_dni = None
        apellido_nombre = cleaned

        order_match = re.search(
            r"\b(?:N[°ºo]?|NRO\.?\s*ORDEN|ORDEN|N\s*ORDEN)\s*[:\-]?\s*(\d{1,6})\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if order_match:
            nro_orden = order_match.group(1)
        else:
            leading_num = re.match(r"^\s*(\d{1,6})\b", cleaned)
            if leading_num:
                nro_orden = leading_num.group(1)

        doc_match = re.search(
            r"\b(?:DNI|DOC(?:UMENTO)?|LE|LC|CI|PASAPORTE)?\s*[:\-]?\s*([\d\.]{6,12})\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if doc_match:
            documento_dni = re.sub(r"\D", "", doc_match.group(1))

        if nro_orden:
            cleaned = re.sub(
                r"\b(?:N[°ºo]?|NRO\.?\s*ORDEN|ORDEN|N\s*ORDEN)\s*[:\-]?\s*\d{1,6}\b",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"^\s*\d{1,6}\b", "", cleaned).strip()

        if documento_dni:
            cleaned = re.sub(
                r"\b(?:DNI|DOC(?:UMENTO)?|LE|LC|CI|PASAPORTE)?\s*[:\-]?\s*[\d\.]{6,12}\b",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

        cleaned = re.sub(r"^[\s,;:\-]+", "", cleaned).strip()
        if cleaned:
            apellido_nombre = cleaned

        return {
            "nro_orden": nro_orden,
            "documento_dni": documento_dni,
            "apellido_nombre": apellido_nombre,
        }


def _get_pipeline() -> DocumentPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = DocumentPipeline()
    return _PIPELINE


def process_document(file_path: str, document_id: str) -> dict[str, Any]:
    """
    Punto de entrada público que consume app.api.routes.
    """
    return _get_pipeline().run(file_path, document_id)


def save_result(result: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{result['document_id']}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


__all__ = ["DocumentPipeline", "process_document", "save_result"]