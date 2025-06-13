from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

class PlayerResponse(BaseModel):
    person_id: int
    first_name: str
    last_name: str
    full_name: Optional[str] = None
    birthdate: Optional[datetime] = None
    last_attended: Optional[str] = None
    country: Optional[str] = None
    height: Optional[float] = None
    body_weight: Optional[float] = None
    primary_position: Optional[str] = None
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None
    draft_number: Optional[int] = None
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm(cls, player):
        full_name = f"{player.first_name} {player.last_name}"
        position = "Guard" if player.guard else "Forward" if player.forward else "Center" if player.center else "Unknown"
        
        return cls(
            person_id=player.person_id,
            first_name=player.first_name,
            last_name=player.last_name,
            full_name=full_name,
            birthdate=player.birthdate,
            last_attended=player.last_attended,
            country=player.country,
            height=player.height,
            body_weight=player.body_weight,
            primary_position=position,
            draft_year=player.draft_year,
            draft_round=player.draft_round,
            draft_number=player.draft_number
        )

class TeamResponse(BaseModel):
    team_id: int
    team_city: str
    team_name: str
    full_name: Optional[str] = None
    team_abbrev: Optional[str] = None
    season_founded: Optional[int] = None
    league: Optional[str] = None
    
    class Config:
        from_attributes = True
    
    @classmethod
    def from_orm(cls, team):
        return cls(
            team_id=team.team_id,
            team_city=team.team_city,
            team_name=team.team_name,
            full_name=f"{team.team_city} {team.team_name}",
            team_abbrev=team.team_abbrev,
            season_founded=team.season_founded,
            league=team.league
        )

class GameResponse(BaseModel):
    game_id: int
    game_date: datetime
    home_team_id: int
    away_team_id: int
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    winner_team_id: Optional[int] = None
    game_type: Optional[str] = None
    attendance: Optional[int] = None
    
    class Config:
        from_attributes = True

class PlayerStatsResponse(BaseModel):
    game_id: int
    game_date: datetime
    team_id: Optional[int] = None
    opponent_team_id: Optional[int] = None
    win: Optional[bool] = None
    home: Optional[bool] = None
    num_minutes: Optional[float] = None
    points: int = 0
    assists: int = 0
    rebounds_total: int = 0
    steals: int = 0
    blocks: int = 0
    field_goals_made: int = 0
    field_goals_attempted: int = 0
    three_pointers_made: int = 0
    three_pointers_attempted: int = 0
    free_throws_made: int = 0
    free_throws_attempted: int = 0
    turnovers: int = 0
    fouls_personal: int = 0
    plus_minus_points: int = 0
    
    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    home_team_id: int = Field(..., description="Home team ID")
    away_team_id: int = Field(..., description="Away team ID")
    game_date: Optional[datetime] = Field(None, description="Game date (optional)")

class PredictionResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    home_win_probability: float = Field(..., ge=0, le=1)
    away_win_probability: float = Field(..., ge=0, le=1)
    predicted_home_score: float
    predicted_away_score: float
    confidence_score: float = Field(..., ge=0, le=1)
    prediction_date: datetime

class PlayerProjectionResponse(BaseModel):
    player_id: int
    projections: Dict[str, float]
    confidence_score: float = Field(..., ge=0, le=1)
    last_updated: datetime
