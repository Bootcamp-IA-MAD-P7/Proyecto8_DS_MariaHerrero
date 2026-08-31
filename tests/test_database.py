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


def test_self_reported_origin_is_persisted(
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

        patient_data = PATIENT_DATA.copy()
        patient_data["origin"] = (
            "self_reported"
        )

        assessment = (
            assessment_repository.create(
                patient.id,
                patient_data,
            )
        )

        assert (
            assessment.origin
            == "self_reported"
        )

    finally:
        db.close()


def test_professional_origin_is_persisted(
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

        patient_data = PATIENT_DATA.copy()
        patient_data["origin"] = (
            "professional"
        )

        assessment = (
            assessment_repository.create(
                patient.id,
                patient_data,
            )
        )

        assert (
            assessment.origin
            == "professional"
        )

    finally:
        db.close()


def test_patient_assessments_can_have_different_origins(
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

        self_reported_data = (
            PATIENT_DATA.copy()
        )
        self_reported_data["origin"] = (
            "self_reported"
        )

        professional_data = (
            PATIENT_DATA.copy()
        )
        professional_data["origin"] = (
            "professional"
        )

        assessment_repository.create(
            patient.id,
            self_reported_data,
        )

        assessment_repository.create(
            patient.id,
            professional_data,
        )

        assessments = (
            assessment_repository
            .get_by_patient(
                patient.id
            )
        )

        assert len(assessments) == 2

        assert (
            assessments[0].origin
            == "self_reported"
        )

        assert (
            assessments[1].origin
            == "professional"
        )

    finally:
        db.close()


def test_historical_prediction_is_traceable(
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

        patient_data = PATIENT_DATA.copy()
        patient_data["origin"] = (
            "professional"
        )

        assessment = (
            assessment_repository.create(
                patient.id,
                patient_data,
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

        assert prediction.created_at is not None

        assert (
            prediction.assessment.created_at
            is not None
        )

        assert (
            prediction.assessment.patient_id
            == patient.id
        )

        assert (
            prediction.assessment.origin
            == "professional"
        )

        assert (
            prediction.assessment.gender
            == PATIENT_DATA["gender"]
        )

        assert (
            prediction.assessment.age
            == PATIENT_DATA["age"]
        )

        assert (
            prediction.assessment.hypertension
            == PATIENT_DATA["hypertension"]
        )

        assert (
            prediction.assessment.heart_disease
            == PATIENT_DATA["heart_disease"]
        )

        assert (
            prediction.assessment.ever_married
            == PATIENT_DATA["ever_married"]
        )

        assert (
            prediction.assessment.work_type
            == PATIENT_DATA["work_type"]
        )

        assert (
            prediction.assessment.residence_type
            == PATIENT_DATA[
                "Residence_type"
            ]
        )

        assert (
            prediction.assessment.avg_glucose_level
            == PATIENT_DATA[
                "avg_glucose_level"
            ]
        )

        assert (
            prediction.assessment.bmi
            == PATIENT_DATA["bmi"]
        )

        assert (
            prediction.assessment.smoking_status
            == PATIENT_DATA[
                "smoking_status"
            ]
        )

        assert prediction.score == 0.2423
        assert prediction.prediction == 1

        assert (
            prediction.model_version.version
            == "logreg_v1"
        )

        assert (
            prediction.model_version.threshold
            == 0.05
        )

        assert (
            prediction.model_version.calibration_method
            == "sigmoid"
        )

    finally:
        db.close()