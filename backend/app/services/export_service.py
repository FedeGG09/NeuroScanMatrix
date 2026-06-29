
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

import pandas as pd

try:
    from app.services.document_registry import get_document_info
except Exception:  # pragma: no cover
    def get_document_info(document_id: str):
        return None

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    bad_chars = ['/', chr(92), ':', '*', '?', '"', '<', '>', '|']
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name.strip() or "document"


def _base_name(result: Dict[str, Any]) -> str:
    doc_id = result.get("document_id")
    info = get_document_info(doc_id) if doc_id else None
    source_file_name = result.get("source_file_name") or (info or {}).get("file_name") or doc_id or "document"
    return _safe_name(Path(source_file_name).stem)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def rows_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    rows = result.get("rows") or []
    if rows:
        normalized = []
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump()
            elif hasattr(row, "dict"):
                row = row.dict()
            normalized.append({k: _json_safe(v) for k, v in dict(row).items()})
        df = pd.DataFrame(normalized)
    else:
        df = pd.DataFrame([
            {
                "document_id": result.get("document_id"),
                "document_type": result.get("document_type"),
                "status": result.get("status", "ok"),
                "note": "No rows extracted",
            }
        ])

    # Alias de columnas amigable y consistente para exportación.
    rename_map = {
        "apellido_nombre": "apellido_y_nombre",
        "anio_nacimiento": "anio_nacimiento",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns and k != v})

    # Asegurar columnas esperadas si existen en la matriz.
    preferred = [
        "page",
        "column",
        "band_index",
        "block_index",
        "nro_orden",
        "apellido_y_nombre",
        "apellido_nombre",
        "domicilio",
        "documento",
        "dni_tipo",
        "anio_nacimiento",
        "voto",
        "voto_score",
        "confidence",
        "bbox",
        "voto_bbox",
        "raw_text",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def export_result_files(result: Dict[str, Any]) -> Dict[str, str]:
    base_name = _base_name(result)
    csv_path = RESULTS_DIR / f"{base_name}.csv"
    xlsx_path = RESULTS_DIR / f"{base_name}.xlsx"

    df = rows_to_dataframe(result)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)

    return {
        "csv_file_name": csv_path.name,
        "csv_path": str(csv_path),
        "xlsx_file_name": xlsx_path.name,
        "xlsx_path": str(xlsx_path),
    }


def find_export_by_document_id(document_id: str, suffix: str) -> Path | None:
    info = get_document_info(document_id)
    if info and info.get("file_name"):
        base = _safe_name(Path(info["file_name"]).stem)
        candidate = RESULTS_DIR / f"{base}.{suffix}"
        if candidate.exists():
            return candidate

    patterns = [
        f"*{document_id}*.{suffix}",
        f"*.{suffix}",
    ]

    for pattern in patterns:
        matches = sorted(RESULTS_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None
