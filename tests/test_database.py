from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.config import Base
from src.database.repositories import (
    AssessmentRepository,
    ModelVersionRepository,
    PatientRepository,
    PredictionRepository,
)


PATIENT_DATA = {
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


def create_test_session(tmp_path):
    database_path = (
        tmp_path / "test_database.db"
    )

    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={
            "check_same_thread": False,
        },
    )

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )

    return session_factory()


def test_patient_can_be_persisted(
    tmp_path,
):
    db = create_test_session(
        tmp_path
    )

    try:
        repository = PatientRepository(
            db
        )

        patient = repository.create()

        stored_patient = repository.get(
            patient.id
        )

        assert patient.id is not None
        assert stored_patient is not None
        assert (
            stored_patient.id
            == patient.id
        )

    finally:
        db.close()


def test_patient_can_have_multiple_assessments(
    tmp_path,
):
    db = create_test_session(
        tmp_path
    )

    try:
        patient_repository = (
            PatientRepository(db)
        )

        assessment_repository = (
            AssessmentRepository(db)
        )

        patient = (
            patient_repository.create()
        )

        first_data = PATIENT_DATA.copy()

        second_data = (
            PATIENT_DATA.copy()
        )

        second_data["age"] = 68.0
        second_data[
            "avg_glucose_level"
        ] = 180.0

        first_assessment = (
            assessment_repository.create(
                patient.id,
                first_data,
            )
        )

        second_assessment = (
            assessment_repository.create(
                patient.id,
                second_data,
            )
        )

        assessments = (
            assessment_repository
            .get_by_patient(
                patient.id
            )
        )

        assert len(assessments) == 2

        assert (
            first_assessment.id
            != second_assessment.id
        )

        assert assessments[0].age == 67.0
        assert assessments[1].age == 68.0

    finally:
        db.close()


def test_model_version_is_not_duplicated(
    tmp_path,
):
    db = create_test_session(
        tmp_path
    )

    try:
        repository = (
            ModelVersionRepository(db)
        )

        first = repository.get_or_create(
            version="logreg_v1",
            threshold=0.05,
            calibration_method="sigmoid",
        )

        second = repository.get_or_create(
            version="logreg_v1",
            threshold=0.05,
            calibration_method="sigmoid",
        )

        assert first.id == second.id
        assert first.version == "logreg_v1"
        assert first.threshold == 0.05

    finally:
        db.close()


def test_prediction_is_linked_to_assessment_and_model(
    tmp_path,
):
    db = create_test_session(
        tmp_path
    )

    try:
        patient_repository = (
            PatientRepository(db)
        )

        assessment_repository = (
            AssessmentRepository(db)
        )

        model_repository = (
            ModelVersionRepository(db)
        )

        prediction_repository = (
            PredictionRepository(db)
        )

        patient = (
            patient_repository.create()
        )

        assessment = (
            assessment_repository.create(
                patient.id,
                PATIENT_DATA,
            )
        )

        model_version = (
            model_repository.get_or_create(
                version="logreg_v1",
                threshold=0.05,
                calibration_method=(
                    "sigmoid"
                ),
            )
        )

        prediction = (
            prediction_repository.create(
                assessment_id=(
                    assessment.id
                ),
                model_version_id=(
                    model_version.id
                ),
                score=0.2423,
                prediction=1,
            )
        )

        assert (
            prediction.assessment_id
            == assessment.id
        )

        assert (
            prediction.model_version_id
            == model_version.id
        )

        assert (
            prediction.assessment.patient_id
            == patient.id
        )

        assert (
            prediction.model_version.version
            == "logreg_v1"
        )

    finally:
        db.close()


def test_new_assessment_preserves_previous_prediction(
    tmp_path,
):
    db = create_test_session(
        tmp_path
    )

    try:
        patient_repository = (
            PatientRepository(db)
        )

        assessment_repository = (
            AssessmentRepository(db)
        )

        model_repository = (
            ModelVersionRepository(db)
        )

        prediction_repository = (
            PredictionRepository(db)
        )

        patient = (
            patient_repository.create()
        )

        model_version = (
            model_repository.get_or_create(
                version="logreg_v1",
                threshold=0.05,
                calibration_method=(
                    "sigmoid"
                ),
            )
        )

        first_assessment = (
            assessment_repository.create(
                patient.id,
                PATIENT_DATA,
            )
        )

        first_prediction = (
            prediction_repository.create(
                assessment_id=(
                    first_assessment.id
                ),
                model_version_id=(
                    model_version.id
                ),
                score=0.24,
                prediction=1,
            )
        )

        second_data = (
            PATIENT_DATA.copy()
        )

        second_data["age"] = 50.0

        second_assessment = (
            assessment_repository.create(
                patient.id,
                second_data,
            )
        )

        assessments = (
            assessment_repository
            .get_by_patient(
                patient.id
            )
        )

        assert len(assessments) == 2

        assert (
            first_assessment.id
            != second_assessment.id
        )

        assert (
            first_prediction.assessment_id
            == first_assessment.id
        )

        assert (
            first_prediction.score
            == 0.24
        )

    finally:
        db.close()