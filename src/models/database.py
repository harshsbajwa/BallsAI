from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional
import os

Base = declarative_base()

class Player(Base):
    __tablename__ = "players"
    
    person_id = Column(Integer, primary_key=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    birthdate = Column(DateTime)
    last_attended = Column(String(200))
    country = Column(String(100))
    height = Column(Float)
    body_weight = Column(Float)
    guard = Column(Boolean, default=False)
    forward = Column(Boolean, default=False)
    center = Column(Boolean, default=False)
    draft_year = Column(Integer)
    draft_round = Column(Integer)
    draft_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    player_stats = relationship("PlayerStatistic", back_populates="player")

class Team(Base):
    __tablename__ = "teams"
    
    team_id = Column(Integer, primary_key=True)
    team_city = Column(String(100), nullable=False)
    team_name = Column(String(100), nullable=False)
    team_abbrev = Column(String(10))
    season_founded = Column(Integer)
    season_active_till = Column(Integer)
    league = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    team_stats = relationship(
        "TeamStatistic", 
        foreign_keys="[TeamStatistic.team_id]",
        back_populates="team"
    )
    home_games = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")

class Game(Base):
    __tablename__ = "games"
    
    game_id = Column(Integer, primary_key=True)
    game_date = Column(DateTime, nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    winner_team_id = Column(Integer, ForeignKey("teams.team_id"))
    game_type = Column(String(50))
    attendance = Column(Integer)
    arena_id = Column(Integer)
    game_label = Column(String(100))
    game_sub_label = Column(String(100))
    series_game_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    player_stats = relationship("PlayerStatistic", back_populates="game")
    team_stats = relationship("TeamStatistic", back_populates="game")

class PlayerStatistic(Base):
    __tablename__ = "player_statistics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("players.person_id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.game_id"), nullable=False)
    game_date = Column(DateTime, nullable=False)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    opponent_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    win = Column(Boolean)
    home = Column(Boolean)
    num_minutes = Column(Float)
    points = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    blocks = Column(Integer, default=0)
    steals = Column(Integer, default=0)
    field_goals_attempted = Column(Integer, default=0)
    field_goals_made = Column(Integer, default=0)
    field_goals_percentage = Column(Float, default=0.0)
    three_pointers_attempted = Column(Integer, default=0)
    three_pointers_made = Column(Integer, default=0)
    three_pointers_percentage = Column(Float, default=0.0)
    free_throws_attempted = Column(Integer, default=0)
    free_throws_made = Column(Integer, default=0)
    free_throws_percentage = Column(Float, default=0.0)
    rebounds_defensive = Column(Integer, default=0)
    rebounds_offensive = Column(Integer, default=0)
    rebounds_total = Column(Integer, default=0)
    fouls_personal = Column(Integer, default=0)
    turnovers = Column(Integer, default=0)
    plus_minus_points = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    player = relationship("Player", back_populates="player_stats")
    game = relationship("Game", back_populates="player_stats")

class TeamStatistic(Base):
    __tablename__ = "team_statistics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.game_id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    opponent_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    game_date = Column(DateTime, nullable=False)
    home = Column(Boolean)
    win = Column(Boolean)
    team_score = Column(Integer)
    opponent_score = Column(Integer)
    assists = Column(Integer, default=0)
    blocks = Column(Integer, default=0)
    steals = Column(Integer, default=0)
    field_goals_attempted = Column(Integer, default=0)
    field_goals_made = Column(Integer, default=0)
    field_goals_percentage = Column(Float, default=0.0)
    three_pointers_attempted = Column(Integer, default=0)
    three_pointers_made = Column(Integer, default=0)
    three_pointers_percentage = Column(Float, default=0.0)
    free_throws_attempted = Column(Integer, default=0)
    free_throws_made = Column(Integer, default=0)
    free_throws_percentage = Column(Float, default=0.0)
    rebounds_defensive = Column(Integer, default=0)
    rebounds_offensive = Column(Integer, default=0)
    rebounds_total = Column(Integer, default=0)
    fouls_personal = Column(Integer, default=0)
    turnovers = Column(Integer, default=0)
    plus_minus_points = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
   
    team = relationship("Team", foreign_keys=[team_id], back_populates="team_stats")
    opponent_team = relationship("Team", foreign_keys=[opponent_team_id])
    game = relationship("Game", back_populates="team_stats")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres-service")
DATABASE_URL = f"postgresql://postgres:password@{POSTGRES_HOST}:5432/nba_pipeline"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)
