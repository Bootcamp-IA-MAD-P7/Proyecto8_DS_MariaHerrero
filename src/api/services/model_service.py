import json
from pathlib import Path

import joblib

from src.models.final_model import MODEL_VERSION


MODEL_PATH = Path(
    "artifacts/final_model/"
    "stroke_model_logreg_v1.joblib"
)

THRESHOLD_PATH = Path(
    "artifacts/final_model/"
    "threshold_logreg_v1.json"
)


class ModelService:
    def __init__(
        self,
        model_path=MODEL_PATH,
        threshold_path=THRESHOLD_PATH,
    ):
        self.model_path = Path(
            model_path
        )

        self.threshold_path = Path(
            threshold_path
        )

        self.model = None
        self.threshold = None
        self.model_version = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                "No se encontró el modelo final en "
                f"{self.model_path}"
            )

        if not self.threshold_path.exists():
            raise FileNotFoundError(
                "No se encontró la configuración "
                f"del threshold en {self.threshold_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        with open(
            self.threshold_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(
                file
            )

        self.threshold = float(
            metadata["threshold"]
        )

        self.model_version = (
            metadata.get(
                "model_version",
                MODEL_VERSION,
            )
        )

    @property
    def is_loaded(self):
        return (
            self.model is not None
            and self.threshold is not None
        )

    def get_model(self):
        if not self.is_loaded:
            raise RuntimeError(
                "El modelo no está cargado."
            )

        return self.model