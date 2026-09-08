from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from src.database.config import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    assessments: Mapped[list["Assessment"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        nullable=False,
    )

    origin: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="self_reported",
    )

    gender: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    age: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    hypertension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    heart_disease: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    ever_married: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    work_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    residence_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    avg_glucose_level: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    bmi: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    smoking_status: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(
        back_populates="assessments"
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="assessment",
        cascade="all, delete-orphan",
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    version: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
    )

    threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    calibration_method: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="model_version"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    assessment_id: Mapped[int] = mapped_column(
        ForeignKey("assessments.id"),
        nullable=False,
    )

    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"),
        nullable=False,
    )

    score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    prediction: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    assessment: Mapped["Assessment"] = relationship(
        back_populates="predictions"
    )

    model_version: Mapped["ModelVersion"] = relationship(
        back_populates="predictions"
    )