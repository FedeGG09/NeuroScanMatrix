from __future__ import annotations
from pathlib import Path
import fitz
from PIL import Image
import io
import numpy as np

def is_pdf(path: str | Path) -> bool:
    return str(path).lower().endswith(".pdf")

def pdf_to_images(pdf_path: str | Path, max_pages: int = 20, dpi: int = 200) -> list[np.ndarray]:
    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(np.array(img))
    return images
