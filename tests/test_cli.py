from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.api.schemas import PredictionRequest
from src.api.services.model_service import (
    ModelService,
)
from src.api.services.prediction_service import (
    PredictionService,
)
from src.cli.main import (
    DISCLAIMER,
    ask_binary,
    ask_choice,
    ask_float,
    display_result,
)


VALID_PAYLOAD = {
    "origin": "professional",
    "gender": "Female",
    "age": 47,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 99,
    "bmi": 30.1,
    "smoking_status": "smokes",
}


def test_ask_float_accepts_valid_number():
    with patch(
        "builtins.input",
        return_value="47",
    ):
        result = ask_float(
            "Edad"
        )

    assert result == 47.0


def test_ask_float_rejects_invalid_input(
    capsys,
):
    with patch(
        "builtins.input",
        side_effect=[
            "hola",
            "47",
        ],
    ):
        result = ask_float(
            "Edad"
        )

    output = capsys.readouterr().out

    assert (
        "Debes introducir "
        "un número válido."
        in output
    )

    assert result == 47.0


def test_ask_binary_rejects_invalid_input(
    capsys,
):
    with patch(
        "builtins.input",
        side_effect=[
            "5",
            "1",
        ],
    ):
        result = ask_binary(
            "Hipertensión"
        )

    output = capsys.readouterr().out

    assert (
        "Introduce únicamente "
        "0 o 1."
        in output
    )

    assert result == 1


def test_ask_choice_rejects_invalid_option(
    capsys,
):
    with patch(
        "builtins.input",
        side_effect=[
            "8",
            "2",
        ],
    ):
        result = ask_choice(
            "Sexo:",
            [
                "Female",
                "Male",
                "Other",
            ],
        )

    output = capsys.readouterr().out

    assert (
        "Entrada no válida. "
        "Inténtalo de nuevo."
        in output
    )

    assert result == "Male"


def test_display_result_includes_safety_disclaimer(
    capsys,
):
    result = {
        "prediction": 0,
        "score": 0.0288,
        "threshold": 0.05,
        "model_version": "logreg_v1",
    }

    display_result(
        result
    )

    output = capsys.readouterr().out

    assert "Score de riesgo:" in output
    assert "Threshold:" in output
    assert "Clasificación:" in output
    assert "Modelo: logreg_v1" in output

    assert (
        "El score no supera el "
        "umbral configurado."
        in output
    )

    assert DISCLAIMER in output

    assert (
        "no constituye un "
        "diagnóstico médico"
        in output
    )


def test_valid_patient_is_accepted():
    request = PredictionRequest(
        **VALID_PAYLOAD
    )

    assert request.age == 47
    assert request.bmi == 30.1
    assert request.gender == "Female"

    assert (
        request.origin
        == "professional"
    )


def test_missing_required_field_is_rejected():
    payload = VALID_PAYLOAD.copy()

    del payload[
        "avg_glucose_level"
    ]

    with pytest.raises(
        ValidationError
    ):
        PredictionRequest(
            **payload
        )


def test_invalid_age_is_rejected():
    payload = VALID_PAYLOAD.copy()

    payload["age"] = -10

    with pytest.raises(
        ValidationError
    ):
        PredictionRequest(
            **payload
        )


def test_invalid_category_is_rejected():
    payload = VALID_PAYLOAD.copy()

    payload["gender"] = "Invalid"

    with pytest.raises(
        ValidationError
    ):
        PredictionRequest(
            **payload
        )


def test_invalid_bmi_is_rejected():
    payload = VALID_PAYLOAD.copy()

    payload["bmi"] = -5

    with pytest.raises(
        ValidationError
    ):
        PredictionRequest(
            **payload
        )


def test_model_unavailable_fails_controlled():
    class UnavailableModelService:
        is_loaded = False

    service = PredictionService.__new__(
        PredictionService
    )

    service.model_service = (
        UnavailableModelService()
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "El modelo no está "
            "disponible."
        ),
    ):
        service.predict(
            VALID_PAYLOAD
        )


def test_cli_prediction_returns_valid_response():
    model_service = ModelService()
    model_service.load()

    prediction_service = (
        PredictionService(
            model_service
        )
    )

    request = PredictionRequest(
        **VALID_PAYLOAD
    )

    result = (
        prediction_service.predict(
            request
        )
    )

    assert (
        result["prediction"]
        in [0, 1]
    )

    assert (
        0 <= result["score"] <= 1
    )

    assert (
        result["threshold"]
        == 0.05
    )

    assert (
        result["model_version"]
        == "logreg_v1"
    )


def test_cli_prediction_is_consistent_with_api():
    model_service = ModelService()
    model_service.load()

    prediction_service = (
        PredictionService(
            model_service
        )
    )

    cli_request = PredictionRequest(
        **VALID_PAYLOAD
    )

    cli_result = (
        prediction_service.predict(
            cli_request
        )
    )

    with TestClient(app) as client:
        api_response = client.post(
            "/api/v1/predictions",
            json=VALID_PAYLOAD,
        )

    assert (
        api_response.status_code
        == 200
    )

    api_result = (
        api_response.json()
    )

    assert (
        cli_result["prediction"]
        == api_result["prediction"]
    )

    assert (
        cli_result["score"]
        == api_result["score"]
    )

    assert (
        cli_result["threshold"]
        == api_result["threshold"]
    )

    assert (
        cli_result["model_version"]
        == api_result["model_version"]
    )