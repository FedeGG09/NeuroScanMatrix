# 🧠 NeuroScanMatrix — Intelligent Document Understanding Engine

NeuroScan Matrix es un motor avanzado de procesamiento documental que combina múltiples modelos de visión artificial y NLP para transformar imágenes en datos estructurados listos para análisis.

Inspirado en la estética tecnológica de los 90s, el sistema opera como una “matrix” de procesamiento: múltiples modelos trabajan en conjunto para interpretar, clasificar y estructurar documentos complejos.

# ⚙️ ¿Qué hace?

El sistema va más allá del OCR tradicional:

Carga de documentos e imágenes
Soporte para documentos escaneados y fotos.
Pipeline inteligente multimodelo
OCR de texto impreso
Reconocimiento de escritura manuscrita
Detección de elementos (checkboxes)
Clasificación de documentos
Análisis de layout
Extracción estructurada y tabular
Orquestación automática
Un pipeline central coordina todos los módulos de forma eficiente.
Revisión humana
Interfaz para validar y corregir resultados.
Exportación
Datos estructurados listos para integraciones o análisis.

#🧬 Arquitectura

## El sistema está compuesto por módulos especializados:

## 🧠 OCR híbrido (PaddleOCR + TrOCR)
## 👁️ Computer Vision (YOLOv8)
## 🧩 Layout Understanding (LayoutLMv3)
## 📊 Extracción estructurada y tabular
##🧪 Validación de datos
## 🗂️ Registro y gestión de documentos
## ⚙️ Pipeline central orquestado
##⚡ Características clave
Lazy loading de modelos (arranque liviano)
Fallback heurístico si faltan modelos
Arquitectura modular y extensible
Preparado para fine-tuning por dominio
Pipeline robusto tipo Document AI

# 🎮 Filosofía

"See beyond the pixels."

NeuroScan Matrix no solo lee documentos:
los interpreta, los estructura y los convierte en datos accionables.

# 🚀 Casos de uso
Formularios
Facturas
Documentación administrativa
Digitalización de archivos históricos
Automatización documental


# OCR Advanced System V4

Stack:
- FastAPI backend
- PaddleOCR for printed OCR
- TrOCR for handwritten text
- YOLOv8 for checkbox detection
- LayoutLMv3 for structured extraction
- Simple frontend for human review

## Run backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Open frontend

Open `frontend/index.html` directly or serve it with any static server.

## API

- `POST /upload`
- `POST /extract`
- `GET /documents/{document_id}`
- `POST /documents/{document_id}/review`
- `GET /files/{document_id}`

