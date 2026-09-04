from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": None,
                "origin": "self_reported",
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
        }
    )

    patient_id: int | None = Field(
        default=None,
        gt=0,
    )

    origin: Literal[
        "professional",
        "self_reported",
    ] = "self_reported"

    gender: Literal[
        "Female",
        "Male",
        "Other",
    ]

    age: float = Field(
        ...,
        ge=0,
        le=120,
        allow_inf_nan=False,
    )

    hypertension: Literal[0, 1]

    heart_disease: Literal[0, 1]

    ever_married: Literal[
        "Yes",
        "No",
    ]

    work_type: Literal[
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked",
    ]

    Residence_type: Literal[
        "Urban",
        "Rural",
    ]

    avg_glucose_level: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
    )

    bmi: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
    )

    smoking_status: Literal[
        "formerly smoked",
        "never smoked",
        "smokes",
        "Unknown",
    ]


class InfluenceFactor(BaseModel):
    feature: str
    value: str | int | float
    reference_value: str | int | float
    influence: float


class ExplanationResponse(BaseModel):
    model_version: str
    score: float
    threshold: float
    prediction: int

    factors_increasing_score: list[
        InfluenceFactor
    ]

    factors_decreasing_score: list[
        InfluenceFactor
    ]

    interpretation: str
    disclaimer: str


class PredictionResponse(BaseModel):
    patient_id: int
    assessment_id: int
    prediction_id: int

    prediction: int
    score: float
    threshold: float
    model_version: str

    explanation: ExplanationResponse


class AssessmentHistoryItem(BaseModel):
    assessment_id: int
    patient_id: int
    assessment_created_at: datetime
    prediction_created_at: datetime | None
    origin: str
    prediction: int | None
    score: float | None
    threshold: float | None
    model_version: str | None


class AssessmentHistoryDetail(
    AssessmentHistoryItem
):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ErrorResponse(BaseModel):
    detail: str
