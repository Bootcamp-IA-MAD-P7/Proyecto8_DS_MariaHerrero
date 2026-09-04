import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.services.prediction_service import PredictionService
from src.database.config import Base
from src.database.models import Assessment, Prediction
from src.database.repositories import PatientRepository


PATIENT_DATA = {
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
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, dataframe):
        return np.array(
            [
                [1 - self.probability, self.probability]
                for _ in range(len(dataframe))
            ]
        )


class FakeModelService:
    is_loaded = True
    threshold = 0.5
    model_version = "test_v1"

    def __init__(self, probability):
        self.model = FixedProbabilityModel(probability)

    def get_model(self):
        return self.model


@pytest.fixture
def prediction_dependencies(tmp_path):
    database_path = tmp_path / "predictions.db"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    train_path = tmp_path / "train.csv"
    training_data = pd.DataFrame(
        [
            {**PATIENT_DATA, "stroke": 0},
            {
                **PATIENT_DATA,
                "gender": "Male",
                "age": 45.0,
                "stroke": 1,
            },
        ]
    ).drop(columns=["origin"])
    training_data.to_csv(train_path, index=False)

    try:
        yield session_factory, train_path
    finally:
        engine.dispose()


def create_service(
    probability,
    session_factory,
    train_path,
):
    return PredictionService(
        FakeModelService(probability),
        train_path=train_path,
        session_factory=session_factory,
    )


@pytest.mark.parametrize(
    ("score", "expected_prediction"),
    [
        (0.49, 0),
        (0.50, 1),
        (0.51, 1),
    ],
)
def test_prediction_respects_threshold_boundary(
    prediction_dependencies,
    score,
    expected_prediction,
):
    session_factory, train_path = prediction_dependencies
    service = create_service(
        score,
        session_factory,
        train_path,
    )

    result = service.predict(PATIENT_DATA)

    assert result["score"] == score
    assert result["prediction"] == expected_prediction
    assert result["threshold"] == 0.5
    assert result["model_version"] == "test_v1"


def test_prediction_creates_patient_and_persists_origin(
    prediction_dependencies,
):
    session_factory, train_path = prediction_dependencies
    service = create_service(
        0.75,
        session_factory,
        train_path,
    )

    result = service.predict(PATIENT_DATA)

    db = session_factory()
    try:
        assessment = db.get(
            Assessment,
            result["assessment_id"],
        )
        prediction = db.get(
            Prediction,
            result["prediction_id"],
        )

        assert result["patient_id"] == assessment.patient_id
        assert assessment.origin == "professional"
        assert prediction.score == 0.75
        assert prediction.prediction == 1
    finally:
        db.close()


def test_prediction_reuses_existing_patient(
    prediction_dependencies,
):
    session_factory, train_path = prediction_dependencies
    db = session_factory()
    try:
        patient = PatientRepository(db).create()
        patient_id = patient.id
    finally:
        db.close()
    service = create_service(
        0.25,
        session_factory,
        train_path,
    )
    patient_data = {
        **PATIENT_DATA,
        "patient_id": patient_id,
    }

    result = service.predict(patient_data)

    assert result["patient_id"] == patient_id
    db = session_factory()
    try:
        assessment = db.get(
            Assessment,
            result["assessment_id"],
        )
        assert assessment.patient_id == patient_id
        assert assessment.origin == "professional"
    finally:
        db.close()


def test_prediction_rejects_unknown_patient_id(
    prediction_dependencies,
):
    session_factory, train_path = prediction_dependencies
    service = create_service(
        0.75,
        session_factory,
        train_path,
    )
    patient_data = {
        **PATIENT_DATA,
        "patient_id": 999,
    }

    with pytest.raises(
        ValueError,
        match="No existe un paciente con id 999",
    ):
        service.predict(patient_data)
