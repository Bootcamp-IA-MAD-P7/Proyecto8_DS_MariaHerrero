import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.services.prediction_service import (
    PredictionService as RealPredictionService,
)


PAYLOAD = {
    "origin": "professional",
    "gender": "Female",
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}


class FixedProbabilityModel:
    def predict_proba(self, dataframe):
        probability = 0.42

        return np.array(
            [
                [1 - probability, probability]
                for _ in range(len(dataframe))
            ]
        )


class LoadedModelService:
    is_loaded = True
    threshold = 0.30
    model_version = "integration_v1"

    def load(self):
        return None

    def get_model(self):
        return FixedProbabilityModel()


def test_prediction_is_persisted_and_available_in_history(
    isolated_api_database,
    monkeypatch,
    tmp_path,
):
    train_path = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {**PAYLOAD, "stroke": 0},
            {
                **PAYLOAD,
                "gender": "Male",
                "age": 45.0,
                "stroke": 1,
            },
        ]
    ).drop(columns=["origin"]).to_csv(
        train_path,
        index=False,
    )
    model_service = LoadedModelService()

    def create_prediction_service(
        service,
        session_factory,
    ):
        return RealPredictionService(
            service,
            train_path=train_path,
            session_factory=session_factory,
        )

    monkeypatch.setattr(
        api_main,
        "model_service",
        model_service,
    )
    monkeypatch.setattr(
        api_main,
        "PredictionService",
        create_prediction_service,
    )
    monkeypatch.setattr(
        api_main,
        "prediction_service",
        None,
    )

    with TestClient(api_main.app) as client:
        prediction_response = client.post(
            "/api/v1/predictions",
            json=PAYLOAD,
        )
        assert prediction_response.status_code == 200
        prediction = prediction_response.json()

        history_response = client.get(
            "/api/v1/assessments"
        )
        detail_response = client.get(
            "/api/v1/assessments/"
            f"{prediction['assessment_id']}"
        )

    assert history_response.status_code == 200
    history = history_response.json()
    assert len(history) == 1
    history_item = history[0]

    assert detail_response.status_code == 200
    detail = detail_response.json()

    for stored_result in (history_item, detail):
        assert stored_result["assessment_id"] == (
            prediction["assessment_id"]
        )
        assert stored_result["patient_id"] == (
            prediction["patient_id"]
        )
        assert stored_result["score"] == prediction["score"] == 0.42
        assert stored_result["prediction"] == prediction["prediction"]
        assert prediction["prediction"] == 1
        assert stored_result["threshold"] == prediction["threshold"]
        assert prediction["threshold"] == 0.30
        assert stored_result["model_version"] == (
            prediction["model_version"]
        )
        assert prediction["model_version"] == "integration_v1"
        assert stored_result["origin"] == PAYLOAD["origin"]

    for field in (
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
    ):
        assert detail[field] == PAYLOAD[field]
