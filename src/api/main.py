from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

from src.api.services.model_service import (
    ModelService,
)

from src.api.services.prediction_service import (
    PredictionService,
)


model_service = ModelService()
prediction_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_service

    try:
        model_service.load()

        prediction_service = (
            PredictionService(
                model_service
            )
        )

    except Exception as exc:
        print(
            "Error cargando el modelo:",
            exc,
        )

    yield


app = FastAPI(
    title="Stroke Risk Prediction API",
    description=(
        "API para estimación de riesgo de "
        "stroke como herramienta de apoyo "
        "a la criba. "
        "Las predicciones no constituyen "
        "un diagnóstico médico."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
)
def health():
    return {
        "status": (
            "ok"
            if model_service.is_loaded
            else "degraded"
        ),
        "model_loaded": (
            model_service.is_loaded
        ),
        "model_version": (
            model_service.model_version
        ),
    }


@app.post(
    "/api/v1/predictions",
    response_model=PredictionResponse,
    responses={
        422: {
            "model": ErrorResponse,
            "description": (
                "Datos de entrada inválidos"
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "Modelo no disponible"
            ),
        },
        500: {
            "model": ErrorResponse,
            "description": (
                "Error durante la predicción"
            ),
        },
    },
    tags=["Predictions"],
)
def create_prediction(
    request: PredictionRequest,
):
    if (
        not model_service.is_loaded
        or prediction_service is None
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo de predicción "
                "no está disponible."
            ),
        )

    try:
        return prediction_service.predict(
            request
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "No se pudo generar "
                "la predicción."
            ),
        ) from exc