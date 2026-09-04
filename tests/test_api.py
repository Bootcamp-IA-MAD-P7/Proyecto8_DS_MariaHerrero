import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.clinical_safety import (
    CLINICAL_DISCLAIMER,
    EXPLAINABILITY_DISCLAIMER,
)
from src.database.repositories import (
    AssessmentRepository,
    ModelVersionRepository,
    PatientRepository,
    PredictionRepository,
)


pytestmark = pytest.mark.usefixtures(
    "isolated_api_database"
)


VALID_PAYLOAD = {
    "gender": "Female",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}


@pytest.fixture
def history_api(isolated_api_database):
    with TestClient(app) as client:
        yield client, isolated_api_database


def create_historical_assessment(
    session_factory,
    *,
    created_at,
    score,
    age=67,
):
    db = session_factory()

    try:
        patient = PatientRepository(db).create()
        assessment_data = VALID_PAYLOAD.copy()
        assessment_data["age"] = age
        assessment = AssessmentRepository(db).create(
            patient.id,
            assessment_data,
        )
        assessment.created_at = created_at

        model_version = (
            ModelVersionRepository(db)
            .get_or_create(
                version="logreg_v1",
                threshold=0.05,
                calibration_method="sigmoid",
            )
        )
        prediction = PredictionRepository(db).create(
            assessment_id=assessment.id,
            model_version_id=model_version.id,
            score=score,
            prediction=int(score >= 0.05),
        )
        prediction.created_at = created_at
        db.commit()

        return assessment.id
    finally:
        db.close()


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "logreg_v1"


def test_prediction_endpoint_returns_valid_response():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            json=VALID_PAYLOAD,
        )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0 <= data["score"] <= 1
    assert data["threshold"] == 0.05
    assert data["model_version"] == "logreg_v1"


def test_prediction_contains_explanation():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            json=VALID_PAYLOAD,
        )

    assert response.status_code == 200

    explanation = response.json()[
        "explanation"
    ]

    assert "factors_increasing_score" in explanation
    assert "factors_decreasing_score" in explanation
    assert "interpretation" in explanation
    assert explanation["disclaimer"] == (
        EXPLAINABILITY_DISCLAIMER
    )


def test_api_uses_centralized_clinical_disclaimer():
    assert app.description == CLINICAL_DISCLAIMER


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("age", float("nan")),
        ("age", float("inf")),
        ("age", float("-inf")),
        ("avg_glucose_level", float("nan")),
        ("avg_glucose_level", float("inf")),
        ("avg_glucose_level", float("-inf")),
        ("bmi", float("nan")),
        ("bmi", float("inf")),
        ("bmi", float("-inf")),
    ],
)
def test_non_finite_numeric_input_returns_422(
    field,
    value,
):
    payload = VALID_PAYLOAD.copy()
    payload[field] = value

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            content=json.dumps(payload),
            headers={
                "Content-Type": "application/json",
            },
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["type"] == "finite_number"
    assert detail[0]["input"] in {
        "NaN",
        "Infinity",
        "-Infinity",
    }


def test_invalid_age_returns_422():
    invalid_payload = (
        VALID_PAYLOAD.copy()
    )

    invalid_payload["age"] = -10

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            json=invalid_payload,
        )

    assert response.status_code == 422


def test_invalid_category_returns_422():
    invalid_payload = (
        VALID_PAYLOAD.copy()
    )

    invalid_payload["gender"] = "Invalid"

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            json=invalid_payload,
        )

    assert response.status_code == 422


def test_openapi_contains_required_endpoints():
    with TestClient(app) as client:
        response = client.get(
            "/openapi.json"
        )

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/api/v1/health" in paths
    assert "/api/v1/predictions" in paths
    assert "/api/v1/assessments" in paths
    assert (
        "/api/v1/assessments/{assessment_id}"
        in paths
    )


def test_assessment_history_is_empty(
    history_api,
):
    client, _ = history_api

    response = client.get(
        "/api/v1/assessments"
    )

    assert response.status_code == 200
    assert response.json() == []


def test_assessment_history_is_ordered_and_contains_prediction(
    history_api,
):
    client, session_factory = history_api
    older_id = create_historical_assessment(
        session_factory,
        created_at=datetime(2026, 1, 1),
        score=0.10,
    )
    newer_id = create_historical_assessment(
        session_factory,
        created_at=datetime(2026, 1, 2),
        score=0.20,
    )

    response = client.get(
        "/api/v1/assessments"
    )

    assert response.status_code == 200
    data = response.json()
    assert [
        item["assessment_id"]
        for item in data
    ] == [newer_id, older_id]
    assert data[0]["score"] == 0.20
    assert data[0]["model_version"] == "logreg_v1"
    assert data[0]["threshold"] == 0.05
    assert data[0]["prediction_created_at"] is not None


def test_assessment_history_detail_contains_original_data(
    history_api,
):
    client, session_factory = history_api
    assessment_id = create_historical_assessment(
        session_factory,
        created_at=datetime(2026, 1, 2),
        score=0.20,
        age=72,
    )

    response = client.get(
        f"/api/v1/assessments/{assessment_id}"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["assessment_id"] == assessment_id
    assert data["gender"] == VALID_PAYLOAD["gender"]
    assert data["age"] == 72
    assert data["hypertension"] == VALID_PAYLOAD["hypertension"]
    assert data["heart_disease"] == VALID_PAYLOAD["heart_disease"]
    assert data["ever_married"] == VALID_PAYLOAD["ever_married"]
    assert data["work_type"] == VALID_PAYLOAD["work_type"]
    assert data["Residence_type"] == VALID_PAYLOAD["Residence_type"]
    assert data["avg_glucose_level"] == VALID_PAYLOAD["avg_glucose_level"]
    assert data["bmi"] == VALID_PAYLOAD["bmi"]
    assert data["smoking_status"] == VALID_PAYLOAD["smoking_status"]
    assert data["score"] == 0.20
    assert data["model_version"] == "logreg_v1"


def test_assessment_history_detail_returns_404(
    history_api,
):
    client, _ = history_api

    response = client.get(
        "/api/v1/assessments/999"
    )

    assert response.status_code == 404


def test_assessment_history_uses_latest_prediction(
    history_api,
):
    client, session_factory = history_api
    db = session_factory()

    try:
        patient = PatientRepository(db).create()
        assessment = AssessmentRepository(db).create(
            patient.id,
            VALID_PAYLOAD,
        )
        model_version = (
            ModelVersionRepository(db)
            .get_or_create(
                version="logreg_v1",
                threshold=0.05,
                calibration_method="sigmoid",
            )
        )
        older = PredictionRepository(db).create(
            assessment_id=assessment.id,
            model_version_id=model_version.id,
            score=0.10,
            prediction=1,
        )
        newer = PredictionRepository(db).create(
            assessment_id=assessment.id,
            model_version_id=model_version.id,
            score=0.30,
            prediction=1,
        )
        older.created_at = datetime(2026, 1, 1)
        newer.created_at = datetime(2026, 1, 2)
        db.commit()
        assessment_id = assessment.id
    finally:
        db.close()

    response = client.get(
        f"/api/v1/assessments/{assessment_id}"
    )

    assert response.status_code == 200
    assert response.json()["score"] == 0.30
