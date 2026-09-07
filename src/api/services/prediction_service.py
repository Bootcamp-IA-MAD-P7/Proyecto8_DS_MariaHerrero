import json
from pathlib import Path

import pandas as pd

from src.database.config import (
    SessionLocal,
)
from src.database.repositories import (
    AssessmentRepository,
    ModelVersionRepository,
    PatientRepository,
    PredictionRepository,
)
from src.models.explainability import (
    explanation_for_api,
)
from src.models.final_model import MODEL_VERSION


REFERENCE_VALUES_PATH = Path(
    "artifacts/final_model/"
    "reference_values_logreg_v1.json"
)
EXPECTED_REFERENCE_FEATURES = {
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
}

CALIBRATION_METHOD = "sigmoid"


class PredictionService:
    def __init__(
        self,
        model_service,
        reference_values_path=(
            REFERENCE_VALUES_PATH
        ),
        session_factory=SessionLocal,
    ):
        self.model_service = (
            model_service
        )

        self.session_factory = (
            session_factory
        )

        with open(
            reference_values_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        if metadata.get("model_version") != (
            MODEL_VERSION
        ):
            raise ValueError(
                "La versión de los valores de "
                "referencia no coincide con el modelo."
            )

        reference_values = metadata.get(
            "reference_values"
        )

        if not isinstance(
            reference_values,
            dict,
        ):
            raise ValueError(
                "El artifact no contiene "
                "reference_values válidos."
            )

        if set(reference_values) != (
            EXPECTED_REFERENCE_FEATURES
        ):
            raise ValueError(
                "Las claves de reference_values "
                "no coinciden con las esperadas."
            )

        self.reference_values = reference_values

    def predict(
        self,
        patient_data,
    ):
        if not self.model_service.is_loaded:
            raise RuntimeError(
                "El modelo no está disponible."
            )

        model = (
            self.model_service.get_model()
        )

        if hasattr(
            patient_data,
            "model_dump",
        ):
            patient_data = (
                patient_data.model_dump()
            )

        patient_data = patient_data.copy()

        patient_id = patient_data.pop(
            "patient_id",
            None,
        )

        origin = patient_data.pop(
            "origin",
            "self_reported",
        )

        dataframe = pd.DataFrame(
            [patient_data]
        )

        probability = float(
            model.predict_proba(
                dataframe
            )[0, 1]
        )

        threshold = (
            self.model_service.threshold
        )

        prediction = int(
            probability >= threshold
        )

        explanation = (
            explanation_for_api(
                model=model,
                patient_data=patient_data,
                reference_values=(
                    self.reference_values
                ),
            )
        )

        persistence_data = (
            patient_data.copy()
        )

        persistence_data["origin"] = (
            origin
        )

        persistence = self._persist_prediction(
            patient_id=patient_id,
            patient_data=persistence_data,
            probability=probability,
            prediction=prediction,
            threshold=threshold,
        )

        return {
            "patient_id": (
                persistence["patient_id"]
            ),
            "assessment_id": (
                persistence["assessment_id"]
            ),
            "prediction_id": (
                persistence["prediction_id"]
            ),
            "prediction": prediction,
            "score": probability,
            "threshold": threshold,
            "model_version": (
                self.model_service.model_version
            ),
            "explanation": explanation,
        }

    def _persist_prediction(
        self,
        patient_id,
        patient_data,
        probability,
        prediction,
        threshold,
    ):
        db = self.session_factory()

        try:
            patient_repository = (
                PatientRepository(db)
            )

            assessment_repository = (
                AssessmentRepository(db)
            )

            model_repository = (
                ModelVersionRepository(db)
            )

            prediction_repository = (
                PredictionRepository(db)
            )

            if patient_id is None:
                patient = (
                    patient_repository.create()
                )

            else:
                patient = (
                    patient_repository.get(
                        patient_id
                    )
                )

                if patient is None:
                    raise ValueError(
                        "No existe un paciente "
                        f"con id {patient_id}."
                    )

            assessment = (
                assessment_repository.create(
                    patient_id=patient.id,
                    patient_data=patient_data,
                )
            )

            model_version = (
                model_repository.get_or_create(
                    version=(
                        self.model_service
                        .model_version
                    ),
                    threshold=threshold,
                    calibration_method=(
                        CALIBRATION_METHOD
                    ),
                )
            )

            prediction_record = (
                prediction_repository.create(
                    assessment_id=(
                        assessment.id
                    ),
                    model_version_id=(
                        model_version.id
                    ),
                    score=probability,
                    prediction=prediction,
                )
            )

            return {
                "patient_id": patient.id,
                "assessment_id": (
                    assessment.id
                ),
                "prediction_id": (
                    prediction_record.id
                ),
            }

        finally:
            db.close()
