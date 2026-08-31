from unittest.mock import patch

from fastapi.testclient import TestClient

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
    from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import PredictionRequest
from src.api.services.model_service import (
    ModelService,
)
from src.api.services.prediction_service import (
    PredictionService,
)


def test_cli_prediction_is_consistent_with_api():
    payload = {
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

    model_service = ModelService()
    model_service.load()

    prediction_service = (
        PredictionService(
            model_service
        )
    )

    cli_request = PredictionRequest(
        **payload
    )

    cli_result = (
        prediction_service.predict(
            cli_request
        )
    )

    with TestClient(app) as client:
        api_response = client.post(
            "/api/v1/predictions",
            json=payload,
        )

    assert api_response.status_code == 200

    api_result = api_response.json()

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