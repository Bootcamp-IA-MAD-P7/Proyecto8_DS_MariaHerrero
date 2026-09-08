import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.predictors import Predictor
from src.api.services.prediction_service import PredictionService


REFERENCE_VALUES = {
    "gender": "Female",
    "age": 45.0,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 100.0,
    "bmi": 25.0,
    "smoking_status": "never smoked",
}

VALID_PAYLOAD = {
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


class ReadyModelService:
    is_loaded = True
    model_version = "logreg_v1"


class FakePredictor:
    modality = "experimental"
    model_name = "fake-predictor"
    model_version = "fake_v1"
    is_ready = True

    def __init__(self):
        self.received = None

    def predict(self, input_data):
        self.received = input_data
        return {
            "patient_id": 1,
            "assessment_id": 2,
            "prediction_id": 3,
            "prediction": 1,
            "score": 0.75,
            "threshold": 0.50,
            "model_version": self.model_version,
            "explanation": {
                "model_version": self.model_version,
                "score": 0.75,
                "threshold": 0.50,
                "prediction": 1,
                "factors_increasing_score": [],
                "factors_decreasing_score": [],
                "interpretation": "Resultado experimental de prueba.",
                "disclaimer": "No constituye un diagnóstico médico.",
            },
        }


class UnreadyPredictor(FakePredictor):
    is_ready = False


def create_tabular_service(tmp_path):
    reference_path = tmp_path / "reference_values.json"
    reference_path.write_text(
        json.dumps(
            {
                "model_version": "logreg_v1",
                "reference_values": REFERENCE_VALUES,
            }
        ),
        encoding="utf-8",
    )
    return PredictionService(
        ReadyModelService(),
        reference_values_path=reference_path,
    )


def test_prediction_service_satisfies_predictor_protocol(tmp_path):
    service = create_tabular_service(tmp_path)

    assert isinstance(service, Predictor)
    assert service.modality == "tabular"
    assert service.model_name == "stroke-risk-screening-model"
    assert service.model_version == "logreg_v1"
    assert service.is_ready is True


def test_fake_predictor_satisfies_protocol_without_ml_dependencies():
    predictor = FakePredictor()

    assert isinstance(predictor, Predictor)


def test_fastapi_accepts_predictor_dependency_override(
    monkeypatch,
):
    class LifespanModelService:
        is_loaded = True
        model_version = "lifespan_v1"

        def load(self):
            return None

    predictor = FakePredictor()
    monkeypatch.setattr(
        api_main,
        "model_service",
        LifespanModelService(),
    )
    monkeypatch.setattr(
        api_main,
        "PredictionService",
        lambda *_args, **_kwargs: UnreadyPredictor(),
    )
    monkeypatch.setattr(api_main, "active_predictor", None)
    api_main.app.dependency_overrides[api_main.get_predictor] = (
        lambda: predictor
    )
    try:
        with TestClient(api_main.app) as client:
            response = client.post(
                "/api/v1/predictions",
                json=VALID_PAYLOAD,
            )
    finally:
        api_main.app.dependency_overrides.pop(
            api_main.get_predictor,
            None,
        )

    assert response.status_code == 200
    assert response.json()["model_version"] == "fake_v1"
    assert predictor.received.model_dump(exclude_none=True) == VALID_PAYLOAD


@pytest.mark.parametrize("predictor", [None, UnreadyPredictor()])
def test_missing_or_unready_predictor_returns_503(monkeypatch, predictor):
    monkeypatch.setattr(api_main, "active_predictor", predictor)

    with pytest.raises(HTTPException) as error:
        api_main.get_predictor()

    assert error.value.status_code == 503


def test_predictor_layers_remain_independent():
    base_source = Path("src/api/predictors/base.py").read_text(
        encoding="utf-8"
    )
    prediction_source = Path(
        "src/api/services/prediction_service.py"
    ).read_text(encoding="utf-8")
    ct_source = Path("src/models/ct_cnn.py").read_text(encoding="utf-8")

    for forbidden in ("tensorflow", "keras", "ct_cnn"):
        assert forbidden not in base_source.lower()
    assert "ct_cnn" not in prediction_source.lower()
    assert "prediction_service" not in ct_source.lower()
    assert "artifacts/final_model" in prediction_source
    assert "artifacts/ct_cnn" in ct_source
