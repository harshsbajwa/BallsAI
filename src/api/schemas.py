from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, Dict, Any, List, Generic, TypeVar

# Generic type for paginated responses
DataType = TypeVar("DataType")

class PaginatedResponse(BaseModel, Generic[DataType]):
    """Generic schema for paginated responses."""
    total: int = Field(description="Total number of items available.")
    limit: int = Field(description="The number of items returned in this response.")
    offset: int = Field(description="The starting offset for the returned items.")
    items: List[DataType]

class PlayerResponse(BaseModel):
    """Detailed information about a single player."""
    model_config = ConfigDict(from_attributes=True)
    person_id: int
    full_name: str
    primary_position: Optional[str]
    height: Optional[float] = None
    body_weight: Optional[float] = None
    birthdate: Optional[datetime] = None
    country: Optional[str] = None
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None
    draft_number: Optional[int] = None

class TeamResponse(BaseModel):
    """Detailed information about a single team."""
    model_config = ConfigDict(from_attributes=True)
    team_id: int
    full_name: str
    team_abbrev: Optional[str] = None
    team_city: str
    team_name: str

class GameResponse(BaseModel):
    """Detailed information about a single game, including team details."""
    model_config = ConfigDict(from_attributes=True)
    game_id: int
    game_date: datetime
    home_team: TeamResponse
    away_team: TeamResponse
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    winning_team_id: Optional[int] = None

class PlayerStatsResponse(BaseModel):
    """Statistics for a player in a single game."""
    model_config = ConfigDict(from_attributes=True)
    game_id: int
    game_date: datetime
    points: int
    assists: int
    rebounds_total: int
    steals: int
    blocks: int
    turnovers: int
    plus_minus_points: int

class TeamRosterResponse(BaseModel):
    """A team's roster for a given season."""
    team: TeamResponse
    roster: List[PlayerResponse]

class PredictionRequest(BaseModel):
    home_team_id: int = Field(..., description="Home team ID")
    away_team_id: int = Field(..., description="Away team ID")

class PredictionResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    home_win_probability: float = Field(..., ge=0, le=1)
    predicted_home_score: float
    predicted_away_score: float
    confidence_score: float = Field(..., ge=0, le=1)

class PlayerProjectionRequest(BaseModel):
    player_id: int = Field(..., description="Player ID")
    opponent_team_id: int = Field(..., description="Opponent team ID")
    home_game: bool = Field(..., description="Is it a home game for the player?")

class PlayerImpactRatingResponse(BaseModel):
    """Represents a player's calculated impact rating for a season."""
    model_config = ConfigDict(from_attributes=True)
    person_id: int
    full_name: str
    season_year: int
    primary_position: Optional[str]
    player_impact_rating: float
    avg_ts_percentage: float

class PlayerProjectionResponse(BaseModel):
    player_id: int
    projections: Dict[str, float]
    confidence_score: float = Field(..., ge=0, le=1)

class LeagueLeaderResponse(BaseModel):
    """Represents a single player in the league leader rankings."""
    model_config = ConfigDict(from_attributes=True)
    player: PlayerResponse
    team: TeamResponse
    games_played: int
    avg_points: float
    avg_assists: float
    avg_rebounds: float

class HeadToHeadResponse(BaseModel):
    """Analytics for head-to-head matchups between two teams."""
    model_config = ConfigDict(from_attributes=True)
    team_1: TeamResponse
    team_2: TeamResponse
    team_1_wins: int
    team_2_wins: int
    total_games: int
    avg_margin: float
    last_meeting: datetime