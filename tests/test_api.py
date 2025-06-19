import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.models import database as db_models


@pytest.fixture(scope="function")
def seed_data(test_db: Session):
    """Pre-populate the test database with some known data."""
    player1 = db_models.Player(
        person_id=2544, first_name="LeBron", last_name="James"
    )
    player2 = db_models.Player(
        person_id=201939, first_name="Stephen", last_name="Curry"
    )
    team1 = db_models.Team(
        team_id=1610612738, team_city="Boston", team_name="Celtics"
    )
    test_db.add_all([player1, player2, team1])
    test_db.commit()
    return {"players": [player1, player2], "team": team1}


def test_health_check(client: TestClient):
    """Test: The health check endpoint should always return 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "database": "healthy"}


def test_search_players_happy_path(client: TestClient, seed_data):
    """Test: Searching for a player that exists."""
    response = client.get("/api/v1/players/search?q=LeBron")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["full_name"] == "LeBron James"
    assert data["items"][0]["person_id"] == 2544


def test_get_player_not_found(client: TestClient):
    """Test: Requesting a player ID that does not exist should return 404."""
    response = client.get("/api/v1/players/99999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Player not found"}


def test_search_players_validation_error(client: TestClient):
    """Test: A search query that is too short should return a 422 error."""
    response = client.get("/api/v1/players/search?q=L")
    assert response.status_code == 422
    assert "String should have at least 2 characters" in str(
        response.json()
    )


def test_get_teams(client: TestClient, seed_data):
    """Test: The get teams endpoint should return the seeded team."""
    response = client.get("/api/v1/teams")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert data["items"][0]["full_name"] == "Boston Celtics"


def test_predict_game_mocked(client: TestClient, mocker):
    """Test: The prediction endpoint should work correctly by mocking the ML pipeline."""
    # Mock the helper function that hits the DB to avoid real DB calls
    mocker.patch(
        "src.api.main.get_team_features",
        return_value={"avg_points_scored": 115.0},
    )

    # Mock prediction method
    mocker.patch(
        "src.api.main.ml_pipeline.predict_game_outcome",
        return_value={
            "home_win_probability": 0.75,
            "predicted_home_score": 120.0,
            "predicted_away_score": 110.0,
        },
    )

    # Call endpoint
    response = client.post(
        "/api/v1/predictions/game",
        json={"home_team_id": 1, "away_team_id": 2},
    )

    # Assert response is based on mocked values
    assert response.status_code == 200
    data = response.json()
    assert data["home_win_probability"] == 0.75
    assert data["predicted_home_score"] == 120.0
    assert data["confidence_score"] == 0.5  # abs(0.75 - 0.5) * 2