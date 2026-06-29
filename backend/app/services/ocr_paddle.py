from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class OCRToken:
    text: str
    bbox: list[float]
    confidence: float
    page: int


class PaddleOCREngine:
    """
    OCR multi-backend:
      1) PaddleOCR si está instalado
      2) RapidOCR si está instalado
      3) pytesseract si está instalado y el binario existe

    La idea es no depender de un único motor para que el pipeline siga
    produciendo tokens incluso cuando un backend no esté disponible.
    """

    def __init__(self, lang: str = "es", use_angle_cls: bool = True):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr = None
        self._backend = None

    def _lazy_load(self):
        if self._ocr is not None:
            return self._ocr

        # 1) PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=self.use_angle_cls, show_log=False)
            self._backend = "paddleocr"
            return self._ocr
        except Exception:
            pass

        # 2) RapidOCR
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            self._backend = "rapidocr"
            return self._ocr
        except Exception:
            pass

        self._ocr = None
        self._backend = None
        return None

    def predict(self, image: np.ndarray, page: int = 0) -> list[OCRToken]:
        ocr = self._lazy_load()
        tokens: list[OCRToken] = []

        # Backend 1/2
        if ocr is not None and self._backend == "paddleocr":
            try:
                result = ocr.ocr(image, cls=self.use_angle_cls)
                if not result:
                    return tokens
                lines = result[0] if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list) else result
                for item in lines or []:
                    try:
                        box, (text, conf) = item
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                        tokens.append(OCRToken(text=str(text).strip(), bbox=bbox, confidence=float(conf), page=page))
                    except Exception:
                        continue
                return tokens
            except Exception:
                pass

        if ocr is not None and self._backend == "rapidocr":
            try:
                # RapidOCR normalmente devuelve (result, elapse)
                result = ocr(image)
                lines = result[0] if isinstance(result, tuple) and len(result) >= 1 else result
                for item in lines or []:
                    try:
                        box, text, conf = item
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                        tokens.append(OCRToken(text=str(text).strip(), bbox=bbox, confidence=float(conf), page=page))
                    except Exception:
                        continue
                return tokens
            except Exception:
                pass

        # 3) pytesseract fallback (si existe binario de Tesseract)
        try:
            import pytesseract
            from pytesseract import Output
            import cv2

            img = image
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            data = pytesseract.image_to_data(img, lang=self.lang if self.lang else "eng", output_type=Output.DICT)
            n = len(data.get("text", []))
            for i in range(n):
                txt = str(data["text"][i]).strip()
                conf = float(data["conf"][i]) if str(data["conf"][i]).strip() not in {"-1", ""} else 0.0
                if not txt:
                    continue
                x = float(data["left"][i])
                y = float(data["top"][i])
                w = float(data["width"][i])
                h = float(data["height"][i])
                tokens.append(OCRToken(text=txt, bbox=[x, y, x + w, y + h], confidence=max(0.0, conf / 100.0), page=page))
            return tokens
        except Exception:
            return tokens
