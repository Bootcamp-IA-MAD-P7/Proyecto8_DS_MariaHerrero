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
    build_reference_values,
    explanation_for_api,
)


TRAIN_PATH = "data/processed/train.csv"
TARGET = "stroke"

CALIBRATION_METHOD = "sigmoid"


class PredictionService:
    def __init__(
        self,
        model_service,
        train_path=TRAIN_PATH,
        session_factory=SessionLocal,
    ):
        self.model_service = (
            model_service
        )

        self.session_factory = (
            session_factory
        )

        train = pd.read_csv(
            train_path
        )

        X_train = train.drop(
            columns=[TARGET]
        )

        self.reference_values = (
            build_reference_values(
                X_train
            )
        )

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

        persistence = self._persist_prediction(
            patient_id=patient_id,
            patient_data=patient_data,
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