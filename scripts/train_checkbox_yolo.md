# Training YOLOv8 checkbox detector

1. Annotate checkbox, checkbox_checked, checkbox_unchecked, x_mark, tick_mark.
2. Export dataset to YOLO format.
3. Train:
   ```bash
   yolo detect train model=yolov8n.pt data=checkboxes.yaml imgsz=1024 epochs=100
   ```
4. Save best weights to `backend/models/checkboxes.pt`
