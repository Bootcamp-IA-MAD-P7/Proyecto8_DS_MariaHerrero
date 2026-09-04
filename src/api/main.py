from contextlib import asynccontextmanager
import math

from fastapi import Depends, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.clinical_safety import (
    CLINICAL_DISCLAIMER,
)
from src.api.schemas import (
    AssessmentHistoryDetail,
    AssessmentHistoryItem,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.database.config import SessionLocal, get_db
from src.database.repositories import (
    AssessmentRepository,
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
                model_service,
                session_factory=(
                    app.state.session_factory
                ),
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
    description=CLINICAL_DISCLAIMER,
    version="1.0.0",
    lifespan=lifespan,
)

app.state.session_factory = SessionLocal


def sanitize_validation_value(value):
    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        if math.isnan(value):
            return "NaN"

        return (
            "Infinity"
            if value > 0
            else "-Infinity"
        )

    if isinstance(value, dict):
        return {
            key: sanitize_validation_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            sanitize_validation_value(item)
            for item in value
        ]

    return value


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request,
    exc,
):
    errors = sanitize_validation_value(
        exc.errors()
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": jsonable_encoder(errors),
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def assessment_history_data(
    assessment,
    repository,
):
    latest_prediction = (
        repository.get_latest_prediction(
            assessment
        )
    )

    if latest_prediction is None:
        prediction_data = {
            "prediction_created_at": None,
            "prediction": None,
            "score": None,
            "threshold": None,
            "model_version": None,
        }
    else:
        prediction_data = {
            "prediction_created_at": (
                latest_prediction.created_at
            ),
            "prediction": (
                latest_prediction.prediction
            ),
            "score": latest_prediction.score,
            "threshold": (
                latest_prediction
                .model_version.threshold
            ),
            "model_version": (
                latest_prediction
                .model_version.version
            ),
        }

    return {
        "assessment_id": assessment.id,
        "patient_id": assessment.patient_id,
        "assessment_created_at": (
            assessment.created_at
        ),
        "origin": assessment.origin,
        **prediction_data,
    }


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


@app.get(
    "/api/v1/assessments",
    response_model=list[AssessmentHistoryItem],
    tags=["Assessments"],
)
def list_assessments(
    db: Session = Depends(get_db),
):
    repository = AssessmentRepository(db)

    return [
        assessment_history_data(
            assessment,
            repository,
        )
        for assessment in repository.list_all()
    ]


@app.get(
    "/api/v1/assessments/{assessment_id}",
    response_model=AssessmentHistoryDetail,
    responses={
        404: {
            "model": ErrorResponse,
            "description": (
                "Evaluación no encontrada"
            ),
        },
    },
    tags=["Assessments"],
)
def get_assessment(
    assessment_id: int,
    db: Session = Depends(get_db),
):
    repository = AssessmentRepository(db)
    assessment = (
        repository.get_with_predictions(
            assessment_id
        )
    )

    if assessment is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No existe una evaluación "
                f"con id {assessment_id}."
            ),
        )

    return {
        **assessment_history_data(
            assessment,
            repository,
        ),
        "gender": assessment.gender,
        "age": assessment.age,
        "hypertension": assessment.hypertension,
        "heart_disease": assessment.heart_disease,
        "ever_married": assessment.ever_married,
        "work_type": assessment.work_type,
        "Residence_type": (
            assessment.residence_type
        ),
        "avg_glucose_level": (
            assessment.avg_glucose_level
        ),
        "bmi": assessment.bmi,
        "smoking_status": assessment.smoking_status,
    }


@app.post(
    "/api/v1/predictions",
    response_model=PredictionResponse,
    responses={
        422: {
            "model": ErrorResponse,
            "description": (
                "Datos de entrada inválidos "
                "o paciente inexistente"
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
