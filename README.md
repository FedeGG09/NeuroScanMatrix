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

## Notes

This version is designed to run even without trained weights:
- PaddleOCR/TrOCR/LayoutLMv3/YOLO are loaded lazily.
- If models are missing, the pipeline falls back to heuristics, so the app stays usable.
- To reach production quality, fine-tune the document classifier, checkbox detector, and LayoutLMv3 extractor on your own documents.
