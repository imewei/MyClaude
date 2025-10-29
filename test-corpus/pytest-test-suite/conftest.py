
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_database():
    """Mock database connection."""
    db = MagicMock()
    db.query.return_value = []
    return db

@pytest.fixture
def sample_data():
    """Sample test data."""
    return {
        "id": 1,
        "name": "Test Item",
        "value": 42
    }

@pytest.fixture(scope="session")
def app():
    """Application instance for testing."""
    from app import create_app
    app = create_app("testing")
    yield app
