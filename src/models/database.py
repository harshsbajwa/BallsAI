from sqlalchemy import (
    create_engine, Column, Integer, DateTime, Float, Boolean, 
    ForeignKey, Text, DDL, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime
import os

# Base for application-managed tables (public schema)
Base = declarative_base()

# Base for all models to unify the class registry
AnalyticsBase = Base

# Application-Managed Models
class Player(Base):
    __tablename__ = "players"
    __table_args__ = {'schema': 'public'}
    
    person_id = Column(Integer, primary_key=True)
    first_name = Column(Text, nullable=False)
    last_name = Column(Text, nullable=False)
    birthdate = Column(DateTime)
    last_attended = Column(Text)
    country = Column(Text)
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
    
    @hybrid_property
    def full_name(self):
        return self.first_name + " " + self.last_name
    
    @hybrid_property
    def primary_position(self):
        if self.guard: return 'Guard'
        if self.forward: return 'Forward'
        if self.center: return 'Center'
        return 'Unknown'

    player_stats = relationship("PlayerStatistic", back_populates="player")

class Team(Base):
    __tablename__ = "teams"
    __table_args__ = {'schema': 'public'}
    
    team_id = Column(Integer, primary_key=True)
    team_city = Column(Text, nullable=False)
    team_name = Column(Text, nullable=False)
    team_abbrev = Column(Text)
    season_founded = Column(Integer)
    season_active_till = Column(Integer)
    league = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @hybrid_property
    def full_name(self):
        return self.team_city + " " + self.team_name

    team_stats = relationship("TeamStatistic", foreign_keys="[TeamStatistic.team_id]", back_populates="team")
    home_games = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")

class Game(Base):
    __tablename__ = "games"
    __table_args__ = {'schema': 'public'}
    
    game_id = Column(Integer, primary_key=True)
    game_date = Column(DateTime, nullable=False)
    home_team_id = Column(Integer, ForeignKey("public.teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("public.teams.team_id"), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    winning_team_id = Column(Integer)
    game_type = Column(Text)
    attendance = Column(Integer)
    arena_id = Column(Integer)
    game_label = Column(Text)
    game_sub_label = Column(Text)
    series_game_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    player_stats = relationship("PlayerStatistic", back_populates="game")
    team_stats = relationship("TeamStatistic", back_populates="game")

class PlayerStatistic(Base):
    __tablename__ = "player_statistics"
    __table_args__ = {'schema': 'public'}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("public.players.person_id"), nullable=False)
    game_id = Column(Integer, ForeignKey("public.games.game_id"), nullable=False)
    game_date = Column(DateTime, nullable=False)
    team_id = Column(Integer, ForeignKey("public.teams.team_id"), nullable=False)
    opponent_team_id = Column(Integer, ForeignKey("public.teams.team_id"), nullable=False)
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
    __table_args__ = {'schema': 'public'}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("public.games.game_id"), nullable=False)
    team_id = Column(Integer, ForeignKey("public.teams.team_id"), nullable=False)
    opponent_team_id = Column(Integer, ForeignKey("public.teams.team_id"), nullable=False)
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

# DBT-Managed Models
class PlayerSeasonStats(AnalyticsBase):
    __tablename__ = "player_season_stats"
    __table_args__ = {'schema': 'marts', 'extend_existing': True}
    
    person_id = Column(Integer, ForeignKey('public.players.person_id'), primary_key=True)
    season_year = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('public.teams.team_id'), primary_key=True)
    full_name = Column(Text)
    team_name = Column(Text)
    games_played = Column(Integer)
    avg_minutes = Column(Float)
    avg_points = Column(Float)
    avg_assists = Column(Float)
    avg_rebounds = Column(Float)
    avg_steals = Column(Float)
    avg_blocks = Column(Float)
    season_fg_percentage = Column(Float)
    season_3p_percentage = Column(Float)
    season_ft_percentage = Column(Float)
    avg_turnovers = Column(Float)
    avg_plus_minus = Column(Float)
    avg_impact_rating = Column(Float)

class PlayerValueAnalytics(AnalyticsBase):
    __tablename__ = "player_value_analytics"
    __table_args__ = {'schema': 'marts', 'extend_existing': True}
    
    person_id = Column(Integer, ForeignKey('public.players.person_id'), primary_key=True)
    season_year = Column(Integer, primary_key=True)
    full_name = Column(Text)
    primary_position = Column(Text)
    draft_year = Column(Integer)
    draft_number = Column(Integer)
    player_impact_rating = Column(Float)
    points_per_36 = Column(Float)
    rebounds_per_36 = Column(Float)
    assists_per_36 = Column(Float)
    avg_ts_percentage = Column(Float)
    calculated_at = Column(DateTime)

class HeadToHeadStats(AnalyticsBase):
    __tablename__ = "head_to_head_stats"
    __table_args__ = {'schema': 'marts', 'extend_existing': True}
    
    team_1_id = Column(Integer, ForeignKey('public.teams.team_id'), primary_key=True)
    team_2_id = Column(Integer, ForeignKey('public.teams.team_id'), primary_key=True)
    team_1_wins = Column(Integer)
    team_2_wins = Column(Integer)
    total_games = Column(Integer)
    avg_margin = Column(Float)
    last_meeting = Column(DateTime)
    calculated_at = Column(DateTime)

# Relationships
Player.player_season_stats = relationship("PlayerSeasonStats", back_populates="player")
PlayerSeasonStats.player = relationship("Player", back_populates="player_season_stats")
PlayerSeasonStats.team = relationship("Team", foreign_keys=[PlayerSeasonStats.team_id])
Team.h2h_stats_as_team1 = relationship("HeadToHeadStats", foreign_keys="[HeadToHeadStats.team_1_id]", back_populates="team_1")
Team.h2h_stats_as_team2 = relationship("HeadToHeadStats", foreign_keys="[HeadToHeadStats.team_2_id]", back_populates="team_2")
HeadToHeadStats.team_1 = relationship("Team", foreign_keys=[HeadToHeadStats.team_1_id], back_populates="h2h_stats_as_team1")
HeadToHeadStats.team_2 = relationship("Team", foreign_keys=[HeadToHeadStats.team_2_id], back_populates="h2h_stats_as_team2")

# Connection
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
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
    """Creates the application-managed tables in the public schema."""
    with engine.connect() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS public"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS staging"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS marts"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS analytics"))
    Base.metadata.create_all(bind=engine)