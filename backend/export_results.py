import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


RESULTS_DIR = Path(r"C:\Users\Usuario\Documents\Repos\ocr_solution\backend\data\results")
OUTPUT_CSV = RESULTS_DIR / "ocr_results.csv"
OUTPUT_XLSX = RESULTS_DIR / "ocr_results.xlsx"


def flatten_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte el bloque 'fields' en columnas planas.
    Ej:
      {"dni": {"value": "123", "confidence": 0.9}}
    -> {"dni_value": "123", "dni_confidence": 0.9}
    """
    flat = {}
    for key, value in (fields or {}).items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f"{key}_{subkey}"] = subvalue
        else:
            flat[key] = value
    return flat


def summarize_visual_elements(visual_elements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resume los elementos visuales para exportación tabular.
    """
    summary = {}

    if not isinstance(visual_elements, dict):
        return summary

    for k, v in visual_elements.items():
        if isinstance(v, list):
            summary[f"{k}_count"] = len(v)
        elif isinstance(v, dict):
            summary[f"{k}_count"] = len(v.keys())
        else:
            summary[f"{k}_count"] = 0

    return summary


def result_to_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte un resultado OCR/IE en una fila plana para CSV/Excel.
    """
    row = {
        "document_id": result.get("document_id"),
        "document_type": result.get("document_type"),
        "confidence_global": result.get("confidence_global"),
        "needs_review": result.get("needs_review"),
        "raw_text": result.get("raw_text", ""),
    }

    row.update(flatten_fields(result.get("fields", {}) or {}))
    row.update(summarize_visual_elements(result.get("visual_elements", {}) or {}))

    return row


def load_json_results_from_folder(folder: Path) -> List[Dict[str, Any]]:
    """
    Lee todos los archivos .json de la carpeta y devuelve una lista de resultados.
    """
    if not folder.exists():
        raise FileNotFoundError(f"No existe la carpeta: {folder}")

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No se encontraron archivos .json en: {folder}")

    results = []
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Si el JSON es un dict, lo agregamos como un solo registro
            if isinstance(data, dict):
                results.append(data)

            # Si el JSON contiene una lista de documentos, la extendemos
            elif isinstance(data, list):
                results.extend([item for item in data if isinstance(item, dict)])

            else:
                print(f"[WARN] Formato no soportado en {file_path.name}, se omite.")

        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON inválido en {file_path.name}: {e}")
        except Exception as e:
            print(f"[ERROR] No se pudo leer {file_path.name}: {e}")

    return results


def export_folder_results(
    folder: Path = RESULTS_DIR,
    output_csv: Path = OUTPUT_CSV,
    output_xlsx: Path = OUTPUT_XLSX,
) -> pd.DataFrame:
    """
    Exporta todos los JSON de la carpeta a CSV y Excel.
    """
    data = load_json_results_from_folder(folder)

    if not data:
        raise ValueError("No se encontraron resultados válidos para exportar.")

    rows = [result_to_row(item) for item in data]
    df = pd.DataFrame(rows)

    preferred_cols = [
        "document_id",
        "document_type",
        "confidence_global",
        "needs_review",
        "raw_text",
    ]
    other_cols = [c for c in df.columns if c not in preferred_cols]
    df = df[[c for c in preferred_cols if c in df.columns] + sorted(other_cols)]

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df.to_excel(output_xlsx, index=False)

    print(f"CSV generado en:  {output_csv}")
    print(f"Excel generado en: {output_xlsx}")

    return df


if __name__ == "__main__":
    df = export_folder_results()
    print(df.head())