from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    app_name: str = "OCR Advanced System V4"
    api_prefix: str = ""
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    uploads_dir: Path = data_dir / "uploads"
    results_dir: Path = data_dir / "results"
    models_dir: Path = Path(__file__).resolve().parents[2] / "models"

    # OCR / model checkpoints
    paddle_lang: str = "es"
    paddle_use_angle_cls: bool = True
    trocr_printed_model: str = "microsoft/trocr-base-printed"
    trocr_handwritten_model: str = "microsoft/trocr-base-handwritten"
    layoutlmv3_model: str = "microsoft/layoutlmv3-base"
    layout_classifier_model: str = "microsoft/layoutlmv3-base"
    yolo_checkbox_weights: str = str(models_dir / "checkboxes.pt")

    # runtime
    max_pages: int = 20
    review_threshold: float = 0.75
    device: str = "cpu"

    class Config:
        env_prefix = "OCR_"

settings = Settings()
