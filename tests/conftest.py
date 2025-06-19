import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from src.api.main import app, get_db
from src.models.database import Base

# Use an in-memory SQLite database for fast, isolated tests
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)


@pytest.fixture(scope="function")
def test_db() -> Generator:
    """Create a fresh, isolated database session for each test."""
    # Remove schema for SQLite before creating tables
    if engine.dialect.name == "sqlite":
        for table in Base.metadata.tables.values():
            table.schema = None

    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db: Session) -> Generator:
    """Create a TestClient for the API."""
    def override_get_db():
        """Dependency override to use the test database."""
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}