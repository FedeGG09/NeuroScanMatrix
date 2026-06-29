
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = DATA_DIR / "documents_index.json"


def _read_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_registry(data: Dict[str, Any]) -> None:
    REGISTRY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def register_document(document_id: str, file_name: str, stored_path: str) -> Dict[str, Any]:
    registry = _read_registry()
    registry[document_id] = {
        "document_id": document_id,
        "file_name": file_name,
        "stored_path": stored_path,
    }
    _write_registry(registry)
    return registry[document_id]


def get_document_info(document_id: str) -> Optional[Dict[str, Any]]:
    return _read_registry().get(document_id)
