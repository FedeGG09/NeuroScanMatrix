from __future__ import annotations

import json

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.pipeline import process_document, save_result
from app.models.schemas import ExtractRequest, ExtractResponse, ReviewPayload, UploadResponse
from app.services.export_service import export_result_files, find_export_by_document_id
from app.utils.file_utils import save_upload

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo inválido")

    document_id, stored_path = await save_upload(file, settings.uploads_dir)
    return UploadResponse(
        document_id=document_id,
        file_name=file.filename,
        stored_path=str(stored_path),
    )


@router.post("/extract", response_model=ExtractResponse)
def extract_document(payload: ExtractRequest):
    candidates = list(settings.uploads_dir.glob(f"{payload.document_id}.*"))
    candidates.extend(list(settings.uploads_dir.glob(f"{payload.document_id}__*.*")))

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No se encontró el archivo cargado para ese document_id",
        )

    file_path = str(candidates[0])
    result = process_document(file_path, payload.document_id)
    save_result(result, settings.results_dir)

    exports = export_result_files(result)
    result["csv_file_name"] = exports["csv_file_name"]
    result["xlsx_file_name"] = exports["xlsx_file_name"]
    result["csv_download_url"] = f"/documents/{payload.document_id}/csv"
    result["xlsx_download_url"] = f"/documents/{payload.document_id}/xlsx"

    return result


@router.get("/documents/{document_id}")
def get_result(document_id: str):
    result_path = settings.results_dir / f"{document_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Resultado no encontrado")
    return json.loads(result_path.read_text(encoding="utf-8"))


@router.get("/documents/{document_id}/csv")
def download_csv(document_id: str):
    file_path = find_export_by_document_id(document_id, "csv")
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="CSV no encontrado")

    return FileResponse(
        str(file_path),
        filename=file_path.name,
        media_type="text/csv",
    )


@router.get("/documents/{document_id}/xlsx")
def download_xlsx(document_id: str):
    file_path = find_export_by_document_id(document_id, "xlsx")
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Excel no encontrado")

    return FileResponse(
        str(file_path),
        filename=file_path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@router.post("/documents/{document_id}/review")
def review_document(document_id: str, payload: ReviewPayload):
    result_path = settings.results_dir / f"{document_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Resultado no encontrado")

    data = json.loads(result_path.read_text(encoding="utf-8"))
    data["reviewed_fields"] = payload.fields
    data["reviewer"] = payload.reviewer
    data["review_notes"] = payload.notes
    data["review_status"] = "human_reviewed"

    result_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {"status": "saved", "document_id": document_id}


@router.get("/files/{document_id}")
def get_file(document_id: str):
    candidates = list(settings.uploads_dir.glob(f"{document_id}.*"))
    candidates.extend(list(settings.uploads_dir.glob(f"{document_id}__*.*")))

    if not candidates:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    return FileResponse(str(candidates[0]))
