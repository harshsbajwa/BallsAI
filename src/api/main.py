import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

from fastapi import FastAPI, Depends, HTTPException, Query, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, and_, or_, desc, text

from src.api import schemas
from src.models import database
from src.models.prediction_model import NBAMLPipeline

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
ml_pipeline = NBAMLPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API service...")
    database.create_tables()
    try:
        ml_pipeline.load_models()
        logger.info("ML models loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not load ML models: {e}. Prediction endpoints will not work.")
    yield
    logger.info("Shutting down API service.")

app = FastAPI(
    title="NBA Analytics API",
    description="Comprehensive NBA data and prediction API",
    version="2.0.0",
    lifespan=lifespan,
)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper Functions
def get_team_features(team_id: int, db: Session) -> Dict:
    stats = db.query(database.TeamStatistic).filter(database.TeamStatistic.team_id == team_id).order_by(desc(database.TeamStatistic.game_date)).limit(10).all()
    if not stats:
        raise HTTPException(status_code=404, detail=f"Not enough data for team ID {team_id}")
    return {
        'avg_points_scored': np.mean([s.team_score for s in stats]),
        'avg_points_allowed': np.mean([s.opponent_score for s in stats]),
        'win_rate': np.mean([1 if s.win else 0 for s in stats]),
        'avg_fg_percentage': np.mean([s.field_goals_percentage for s in stats]),
        'avg_rebounds': np.mean([s.rebounds_total for s in stats]),
        'avg_assists': np.mean([s.assists for s in stats]),
        'avg_turnovers': np.mean([s.turnovers for s in stats]),
    }

def prepare_player_features(recent_games_df: pd.DataFrame, opponent_team_id: int, home_game: bool) -> Dict:
    """Prepare features for player prediction"""
    # TODO: The actual model expects more.
    return {
        'avg_points': recent_games_df['points'].mean(),
        'avg_assists': recent_games_df['assists'].mean(),
        'avg_rebounds': recent_games_df['rebounds_total'].mean(),
        'avg_minutes': recent_games_df['num_minutes'].mean(),
        'fg_percentage': recent_games_df['field_goals_percentage'].mean(),
        'opponent_team_id': opponent_team_id,
        'home_game': 1 if home_game else 0,
        'recent_form': recent_games_df['points'].tail(5).mean(),
    }

# API Routers
router = APIRouter(prefix="/api/v1")
players_router = APIRouter(prefix="/players", tags=["Players"])
teams_router = APIRouter(prefix="/teams", tags=["Teams"])
games_router = APIRouter(prefix="/games", tags=["Games"])
analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])
predictions_router = APIRouter(prefix="/predictions", tags=["Predictions"])

# Player Endpoints
@players_router.get("/search", response_model=schemas.PaginatedResponse[schemas.PlayerResponse])
@limiter.limit("60/minute")
def search_players(request: Request, query: str = Query(..., min_length=2, alias="q"), limit: int = Query(10, ge=1, le=50), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    base_query = db.query(database.Player).filter(database.Player.full_name.ilike(f"%{query}%"))
    total = base_query.count()
    players = base_query.offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "items": players}

@players_router.get("/{player_id}", response_model=schemas.PlayerResponse)
@limiter.limit("120/minute")
def get_player(request: Request, player_id: int, db: Session = Depends(get_db)):
    player = db.query(database.Player).filter(database.Player.person_id == player_id).first()
    if not player: raise HTTPException(status_code=404, detail="Player not found")
    return player

@players_router.get("/{player_id}/stats", response_model=schemas.PaginatedResponse[schemas.PlayerStatsResponse])
@limiter.limit("120/minute")
def get_player_stats(request: Request, player_id: int, limit: int = Query(10, ge=1, le=82), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    base_query = db.query(database.PlayerStatistic).filter(database.PlayerStatistic.person_id == player_id).order_by(desc(database.PlayerStatistic.game_date))
    total = base_query.count()
    stats = base_query.offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "items": stats}

# Team Endpoints
@teams_router.get("/", response_model=schemas.PaginatedResponse[schemas.TeamResponse])
@limiter.limit("60/minute")
def get_teams(request: Request, limit: int = Query(30, ge=1, le=50), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    base_query = db.query(database.Team)
    total = base_query.count()
    teams = base_query.offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "items": teams}

@teams_router.get("/{team_id}/roster", response_model=schemas.TeamRosterResponse)
@limiter.limit("60/minute")
def get_team_roster(request: Request, team_id: int, db: Session = Depends(get_db)):
    team = db.query(database.Team).filter(database.Team.team_id == team_id).first()
    if not team: raise HTTPException(status_code=404, detail="Team not found")
    latest_season = db.query(func.max(database.PlayerStatistic.season_year)).filter(database.PlayerStatistic.team_id == team_id).scalar()
    if not latest_season: raise HTTPException(status_code=404, detail="No roster data found")
    player_ids = db.query(database.PlayerStatistic.person_id).filter(and_(database.PlayerStatistic.team_id == team_id, database.PlayerStatistic.season_year == latest_season)).distinct()
    roster = db.query(database.Player).filter(database.Player.person_id.in_(player_ids)).all()
    return {"team": team, "roster": roster}

# Game Endpoints
@games_router.get("/today", response_model=List[schemas.GameResponse])
@limiter.limit("60/minute")
def get_todays_games(request: Request, db: Session = Depends(get_db)):
    today = datetime.utcnow().date()
    games = db.query(database.Game).options(
        joinedload(database.Game.home_team),
        joinedload(database.Game.away_team)
    ).filter(func.date(database.Game.game_date) == today).all()
    return games

@games_router.get("/{game_id}", response_model=schemas.GameResponse)
@limiter.limit("120/minute")
def get_game(request: Request, game_id: int, db: Session = Depends(get_db)):
    game = db.query(database.Game).options(
        joinedload(database.Game.home_team),
        joinedload(database.Game.away_team)
    ).filter(database.Game.game_id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game

# Analytics Endpoints
@analytics_router.get("/league-leaders", response_model=List[schemas.LeagueLeaderResponse])
@limiter.limit("30/minute")
def get_league_leaders(request: Request, season_year: int = Query(datetime.now().year - 1), db: Session = Depends(get_db)):
    leaders = db.query(database.PlayerSeasonStats).options(joinedload(database.PlayerSeasonStats.player), joinedload(database.PlayerSeasonStats.team)).filter(database.PlayerSeasonStats.season_year == season_year, database.PlayerSeasonStats.games_played > 20).order_by(desc(database.PlayerSeasonStats.avg_points)).limit(20).all()
    return leaders

@analytics_router.get("/h2h", response_model=schemas.HeadToHeadResponse)
@limiter.limit("60/minute")
def get_head_to_head(request: Request, team1_id: int, team2_id: int, db: Session = Depends(get_db)):
    if team1_id == team2_id: raise HTTPException(status_code=400, detail="Team IDs cannot be the same.")
    
    t1_id = min(team1_id, team2_id)
    t2_id = max(team1_id, team2_id)

    h2h_stats = db.query(database.HeadToHeadStats).options(
        joinedload(database.HeadToHeadStats.team_1), 
        joinedload(database.HeadToHeadStats.team_2)
    ).filter(
        and_(
            database.HeadToHeadStats.team_1_id == t1_id, 
            database.HeadToHeadStats.team_2_id == t2_id
        )
    ).first()
    if not h2h_stats: raise HTTPException(status_code=404, detail="No head-to-head record found between these teams.")
    return h2h_stats

@analytics_router.get("/player-impact", response_model=List[schemas.PlayerImpactRatingResponse])
@limiter.limit("30/minute")
def get_player_impact_ratings(
    request: Request, 
    season_year: int = Query(datetime.now().year - 1, description="The season year to query for (e.g., 2024 for 2024-25 season)"),
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db)
):
    leaders = db.query(database.PlayerValueAnalytics)\
        .filter(database.PlayerValueAnalytics.season_year == season_year)\
        .order_by(desc(database.PlayerValueAnalytics.player_impact_rating))\
        .limit(limit)\
        .all()
    if not leaders:
        raise HTTPException(status_code=404, detail=f"No impact rating data found for season {season_year}.")
    return leaders

# Prediction Endpoints
@predictions_router.post("/game", response_model=schemas.PredictionResponse)
@limiter.limit("30/minute")
def predict_game(request: Request, pred_request: schemas.PredictionRequest, db: Session = Depends(get_db)):
    home_features = get_team_features(pred_request.home_team_id, db)
    away_features = get_team_features(pred_request.away_team_id, db)
    prediction = ml_pipeline.predict_game_outcome(home_features, away_features)
    win_prob = prediction["home_win_probability"]
    return {
        "home_team_id": pred_request.home_team_id,
        "away_team_id": pred_request.away_team_id,
        "home_win_probability": win_prob,
        "predicted_home_score": prediction["predicted_home_score"],
        "predicted_away_score": prediction["predicted_away_score"],
        "confidence_score": abs(win_prob - 0.5) * 2,
    }

@predictions_router.post("/player", response_model=schemas.PlayerProjectionResponse)
@limiter.limit("30/minute")
def predict_player_stats(request: Request, pred_request: schemas.PlayerProjectionRequest, db: Session = Depends(get_db)):
    recent_games_df = pd.read_sql(db.query(database.PlayerStatistic).filter(database.PlayerStatistic.person_id == pred_request.player_id).order_by(desc(database.PlayerStatistic.game_date)).limit(15).statement, db.bind)
    if recent_games_df.shape[0] < 5: raise HTTPException(status_code=404, detail="Not enough recent game data for this player to make a prediction.")
    features = prepare_player_features(recent_games_df, pred_request.opponent_team_id, pred_request.home_game)
    projections = ml_pipeline.predict_player_stats(features)
    
    points_std_dev = recent_games_df['points'].std()
    confidence = max(0, 1 - (points_std_dev / 15.0))
    
    return {
        "player_id": pred_request.player_id,
        "projections": projections,
        "confidence_score": confidence,
    }

# Root
@app.get("/", include_in_schema=False)
def root(): return {"message": "NBA Analytics API is running."}

# Health
@app.get("/health", tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "database": "unhealthy"})

# Register Routers
router.include_router(players_router)
router.include_router(teams_router)
router.include_router(games_router)
router.include_router(analytics_router)
router.include_router(predictions_router)
app.include_router(router)