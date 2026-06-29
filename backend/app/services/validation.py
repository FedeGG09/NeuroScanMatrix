from __future__ import annotations
from typing import Any
from app.models.schemas import FieldEvidence

def validate_fields(fields: dict[str, Any], threshold: float = 0.75) -> tuple[float, bool]:
    if not fields:
        return 0.0, True
        
    scores = []
    for field in fields.values():
        try:
            # Intenta leer 'confidence' o 'score' de forma segura
            score_val = getattr(field, "confidence", None) or getattr(field, "score", 0.0)
            scores.append(float(score_val))
        except Exception:
            scores.append(0.0)
            
    if not scores:
        return 0.0, True
        
    global_score = round(sum(scores) / len(scores), 3)
    
    # Validación segura por si algún campo individual está por debajo del umbral
    any_low_confidence = False
    for f in fields.values():
        try:
            score_val = getattr(f, "confidence", None) or getattr(f, "score", 0.0)
            if float(score_val) < threshold:
                any_low_confidence = True
                break
        except Exception:
            any_low_confidence = True
            break
            
    needs_review = global_score < threshold or any_low_confidence
    return global_score, needs_review

def field_to_dict(field) -> dict[str, Any]:
    if field is None:
        return {}
        
    # 1. Búsqueda segura del identificador (evita el AttributeError de .key)
    field_name = (
        getattr(field, "key", None) or 
        getattr(field, "name", None) or 
        getattr(field, "field_name", "campo_desconocido")
    )
    
    # 2. Búsqueda segura del nivel de confianza
    conf_val = getattr(field, "confidence", None) or getattr(field, "score", 0.0)
    try:
        confidence = float(conf_val)
    except Exception:
        confidence = 0.0

    # 3. Retorno mapeado usando getattr para evitar caídas por falta de opcionales
    return {
        "field_name": field_name,
        "value": getattr(field, "value", ""),
        "confidence": confidence,
        "source": getattr(field, "source", "ocr"),
        "page": getattr(field, "page", 0),
        "bbox": getattr(field, "bbox", None),
        "ambiguous": getattr(field, "ambiguous", False),
        "candidates": getattr(field, "candidates", []),
    }