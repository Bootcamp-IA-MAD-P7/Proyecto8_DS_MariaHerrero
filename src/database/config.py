from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    sessionmaker,
)


DATABASE_PATH = Path("data/stroke_app.db")

DATABASE_URL = (
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)


class Base(DeclarativeBase):
    pass


def create_database_engine(
    database_url=DATABASE_URL,
):
    if database_url.startswith("sqlite"):
        DATABASE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    connect_args = {}

    if database_url.startswith("sqlite"):
        connect_args = {
            "check_same_thread": False,
        }

    return create_engine(
        database_url,
        connect_args=connect_args,
    )


engine = create_database_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()