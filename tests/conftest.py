import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.main import app
from src.database.config import Base, get_db


@pytest.fixture
def isolated_api_database(tmp_path):
    database_path = tmp_path / "test_api.db"
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

    def override_get_db():
        db = session_factory()

        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = (
        override_get_db
    )
    original_session_factory = (
        app.state.session_factory
    )
    app.state.session_factory = session_factory

    try:
        yield session_factory
    finally:
        app.state.session_factory = (
            original_session_factory
        )
        app.dependency_overrides.pop(
            get_db,
            None,
        )
        engine.dispose()
