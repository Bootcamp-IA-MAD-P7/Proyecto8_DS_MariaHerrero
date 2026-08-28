import pandas as pd

from src.models.explainability import (
    build_reference_values,
    explanation_for_api,
)


TRAIN_PATH = "data/processed/train.csv"
TARGET = "stroke"


class PredictionService:
    def __init__(
        self,
        model_service,
        train_path=TRAIN_PATH,
    ):
        self.model_service = (
            model_service
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

        return {
            "prediction": prediction,
            "score": probability,
            "threshold": threshold,
            "model_version": (
                self.model_service.model_version
            ),
            "explanation": explanation,
        }