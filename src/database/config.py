from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import (
    DeclarativeBase,
    sessionmaker,
)

from configs.settings import DATABASE_URL



class Base(DeclarativeBase):
    pass


def create_database_engine(
    database_url=DATABASE_URL,
):
    if database_url.startswith("sqlite"):
        database_path = make_url(
            database_url
        ).database

        if database_path not in (
            None,
            "",
            ":memory:",
        ):
            Path(database_path).parent.mkdir(
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
