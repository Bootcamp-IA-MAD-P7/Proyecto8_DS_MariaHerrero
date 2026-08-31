from sqlalchemy import select

from src.database.models import (
    Assessment,
    ModelVersion,
    Patient,
    Prediction,
)


class PatientRepository:
    def __init__(self, db):
        self.db = db

    def create(self):
        patient = Patient()

        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)

        return patient

    def get(self, patient_id):
        return self.db.get(
            Patient,
            patient_id,
        )


class AssessmentRepository:
    def __init__(self, db):
        self.db = db

    def create(
        self,
        patient_id,
        patient_data,
    ):
        assessment = Assessment(
            patient_id=patient_id,
            origin=patient_data.get(
                "origin",
                "self_reported",
            ),
            gender=patient_data["gender"],
            age=patient_data["age"],
            hypertension=(
                patient_data["hypertension"]
            ),
            heart_disease=(
                patient_data["heart_disease"]
            ),
            ever_married=(
                patient_data["ever_married"]
            ),
            work_type=(
                patient_data["work_type"]
            ),
            residence_type=(
                patient_data["Residence_type"]
            ),
            avg_glucose_level=(
                patient_data[
                    "avg_glucose_level"
                ]
            ),
            bmi=patient_data["bmi"],
            smoking_status=(
                patient_data["smoking_status"]
            ),
        )

        self.db.add(assessment)
        self.db.commit()
        self.db.refresh(assessment)

        return assessment

    def get_by_patient(
        self,
        patient_id,
    ):
        statement = (
            select(Assessment)
            .where(
                Assessment.patient_id
                == patient_id
            )
            .order_by(
                Assessment.created_at
            )
        )

        return list(
            self.db.scalars(
                statement
            ).all()
        )


class ModelVersionRepository:
    def __init__(self, db):
        self.db = db

    def get_or_create(
        self,
        version,
        threshold,
        calibration_method,
    ):
        statement = select(
            ModelVersion
        ).where(
            ModelVersion.version
            == version
        )

        model_version = (
            self.db.scalar(statement)
        )

        if model_version is not None:
            return model_version

        model_version = ModelVersion(
            version=version,
            threshold=threshold,
            calibration_method=(
                calibration_method
            ),
        )

        self.db.add(model_version)
        self.db.commit()
        self.db.refresh(model_version)

        return model_version


class PredictionRepository:
    def __init__(self, db):
        self.db = db

    def create(
        self,
        assessment_id,
        model_version_id,
        score,
        prediction,
    ):
        prediction_record = Prediction(
            assessment_id=assessment_id,
            model_version_id=(
                model_version_id
            ),
            score=score,
            prediction=prediction,
        )

        self.db.add(prediction_record)
        self.db.commit()
        self.db.refresh(
            prediction_record
        )

        return prediction_record