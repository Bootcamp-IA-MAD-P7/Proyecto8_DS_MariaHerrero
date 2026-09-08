import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.services.prediction_service import PredictionService
from src.database.config import Base
from src.database.models import Assessment, Prediction
from src.database.repositories import PatientRepository
from src.models.explainability import (
    build_reference_values,
)
from src.models.generate_reference_values import (
    generate_reference_values_artifact,
)


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
    reference_values_path = (
        tmp_path / "reference_values.json"
    )
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
    generate_reference_values_artifact(
        train_path,
        reference_values_path,
    )

    try:
        yield (
            session_factory,
            train_path,
            reference_values_path,
        )
    finally:
        engine.dispose()


def create_service(
    probability,
    session_factory,
    reference_values_path,
):
    return PredictionService(
        FakeModelService(probability),
        reference_values_path=(
            reference_values_path
        ),
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
    (
        session_factory,
        _train_path,
        reference_values_path,
    ) = prediction_dependencies
    service = create_service(
        score,
        session_factory,
        reference_values_path,
    )

    result = service.predict(PATIENT_DATA)

    assert result["score"] == score
    assert result["prediction"] == expected_prediction
    assert result["threshold"] == 0.5
    assert result["model_version"] == "test_v1"


def test_prediction_creates_patient_and_persists_origin(
    prediction_dependencies,
):
    (
        session_factory,
        _train_path,
        reference_values_path,
    ) = prediction_dependencies
    service = create_service(
        0.75,
        session_factory,
        reference_values_path,
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
    (
        session_factory,
        _train_path,
        reference_values_path,
    ) = prediction_dependencies
    db = session_factory()
    try:
        patient = PatientRepository(db).create()
        patient_id = patient.id
    finally:
        db.close()
    service = create_service(
        0.25,
        session_factory,
        reference_values_path,
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
    (
        session_factory,
        _train_path,
        reference_values_path,
    ) = prediction_dependencies
    service = create_service(
        0.75,
        session_factory,
        reference_values_path,
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


def test_generated_reference_values_match_existing_calculation(
    prediction_dependencies,
):
    (
        _session_factory,
        train_path,
        reference_values_path,
    ) = prediction_dependencies
    train = pd.read_csv(train_path)
    expected = build_reference_values(
        train.drop(columns=["stroke"])
    )
    stored = json.loads(
        reference_values_path.read_text(
            encoding="utf-8"
        )
    )

    assert stored["reference_values"] == {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in expected.items()
    }


def test_reference_values_artifact_must_exist(
    prediction_dependencies,
    tmp_path,
):
    session_factory, _, _ = prediction_dependencies

    with pytest.raises(FileNotFoundError):
        create_service(
            0.5,
            session_factory,
            tmp_path / "missing.json",
        )


@pytest.mark.parametrize(
    ("content", "error_type", "message"),
    [
        (
            "{invalid",
            json.JSONDecodeError,
            None,
        ),
        (
            {"model_version": "logreg_v1"},
            ValueError,
            "reference_values válidos",
        ),
        (
            {
                "model_version": "logreg_v1",
                "reference_values": {"age": 45.0},
            },
            ValueError,
            "claves de reference_values",
        ),
        (
            {
                "model_version": "other_version",
                "reference_values": {},
            },
            ValueError,
            "versión de los valores",
        ),
    ],
)
def test_reference_values_artifact_is_validated(
    prediction_dependencies,
    tmp_path,
    content,
    error_type,
    message,
):
    session_factory, _, _ = prediction_dependencies
    artifact_path = tmp_path / "invalid.json"

    if isinstance(content, str):
        artifact_path.write_text(
            content,
            encoding="utf-8",
        )
    else:
        artifact_path.write_text(
            json.dumps(content),
            encoding="utf-8",
        )

    with pytest.raises(
        error_type,
        match=message,
    ):
        create_service(
            0.5,
            session_factory,
            artifact_path,
        )
