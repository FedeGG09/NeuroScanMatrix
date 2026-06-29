
from __future__ import annotations
from fastapi import UploadFile
from pathlib import Path
import uuid

from app.services.document_registry import register_document


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


async def save_upload(file: UploadFile, uploads_dir: Path, document_id: str | None = None) -> tuple[str, Path]:
    ensure_dir(uploads_dir)
    original_name = Path(file.filename or "document").name
    ext = Path(original_name).suffix.lower() or ".bin"
    stem = Path(original_name).stem
    document_id = document_id or str(uuid.uuid4())
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_", " ") else "_" for ch in stem).strip() or "document"
    dest = uploads_dir / f"{document_id}__{safe_stem}{ext}"
    content = await file.read()
    dest.write_bytes(content)
    register_document(document_id=document_id, file_name=original_name, stored_path=str(dest))
    return document_id, dest
