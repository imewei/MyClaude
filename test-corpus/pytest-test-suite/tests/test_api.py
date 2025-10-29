
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item(sample_data):
    response = client.post("/items/", json=sample_data)
    assert response.status_code == 201
    assert response.json()["name"] == sample_data["name"]

@pytest.mark.parametrize("item_id,expected", [
    (1, 200),
    (999, 404),
    (-1, 400),
])
def test_read_item(item_id, expected):
    response = client.get(f"/items/{item_id}")
    assert response.status_code == expected

def test_database_integration(mock_database):
    # Test with mocked database
    result = mock_database.query("SELECT * FROM items")
    assert result == []
    mock_database.query.assert_called_once()
