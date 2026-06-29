
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    stored_path: str


class ExtractRequest(BaseModel):
    document_id: str


class FieldEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    field_name: Optional[str] = None
    value: Any = None
    confidence: float = 0.0
    source: str = "heuristic"
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    ambiguous: bool = False
    candidates: Optional[List[Any]] = None


class ElectorRow(BaseModel):
    """
    Fila completa del padrón, preparada para exportación tabular.
    """
    model_config = ConfigDict(extra="allow")

    nro_orden: Optional[str] = None
    apellido_nombre: Optional[str] = None
    nombre: Optional[str] = None
    domicilio: Optional[str] = None
    documento: Optional[str] = None
    dni_tipo: Optional[str] = None
    anio_nacimiento: Optional[str] = None
    voto: bool = False
    confidence: float = 0.0
    page: Optional[int] = None
    column: Optional[str] = None
    bbox: Optional[List[float]] = None
    raw_text: Optional[str] = None


class ExtractResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str
    document_type: str
    confidence_global: float
    needs_review: bool
    fields: Dict[str, FieldEvidence] = Field(default_factory=dict)
    raw_text: str = ""
    visual_elements: Dict[str, Any] = Field(default_factory=dict)
    pages_processed: int = 0
    status: str = "ok"
    source_file_name: Optional[str] = None
    ocr_tokens: List[Dict[str, Any]] = Field(default_factory=list)
    rows: List[ElectorRow] = Field(default_factory=list)
    tables: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    csv_file_name: Optional[str] = None
    csv_download_url: Optional[str] = None
    xlsx_file_name: Optional[str] = None
    xlsx_download_url: Optional[str] = None


class ReviewPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    fields: Dict[str, Any]
    reviewer: Optional[str] = None
    notes: Optional[str] = None
