from fastapi.testclient import TestClient

from src.api.main import app


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
    assert "disclaimer" in explanation


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