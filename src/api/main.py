from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import redis
import json
from datetime import datetime, timedelta
import logging
import os

from src.models.database import get_db, Player, Team, Game, PlayerStatistic, TeamStatistic
from src.models.prediction_model import NBAMLPipeline
from src.utils.nba_client import NBADataClient
from src.api.schemas import (
    PlayerResponse, TeamResponse, GameResponse, PlayerStatsResponse,
    PredictionRequest, PredictionResponse, PlayerProjectionResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NBA Analytics API",
    description="Comprehensive NBA data and prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = redis.from_url(
    os.getenv("UPSTASH_REDIS_URL", "redis://localhost:6379"),
    decode_responses=True
)

ml_pipeline = NBAMLPipeline()
nba_client = NBADataClient()

@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    try:
        ml_pipeline.load_models()
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")

def get_cache_key(prefix: str, **kwargs) -> str:
    """Generate cache key"""
    key_parts = [prefix]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    return ":".join(key_parts)

def cache_response(key: str, data: Any, ttl: int = 300):
    """Cache response data"""
    try:
        redis_client.setex(key, ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.warning(f"Failed to cache data: {e}")

def get_cached_response(key: str) -> Optional[Dict]:
    """Get cached response"""
    try:
        cached_data = redis_client.get(key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        logger.warning(f"Failed to get cached data: {e}")
    return None

@app.get("/")
async def root():
    return {"message": "NBA Analytics API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_status = "healthy"
        try:
            db = next(get_db())
            db.execute("SELECT 1")
            db.close()
        except Exception:
            db_status = "unhealthy"
        
        redis_status = "healthy"
        try:
            redis_client.ping()
        except Exception:
            redis_status = "unhealthy"
        
        return {
            "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
            "database": db_status,
            "redis": redis_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/players/search", response_model=List[PlayerResponse])
async def search_players(
    q: str = Query(..., min_length=2, description="Player name to search"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Search for players by name"""
    cache_key = get_cache_key("player_search", q=q, limit=limit)
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return cached_result
    
    try:
        players = db.query(Player).filter(
            (Player.first_name.ilike(f"%{q}%")) | 
            (Player.last_name.ilike(f"%{q}%"))
        ).limit(limit).all()
        
        result = [PlayerResponse.from_orm(player) for player in players]
        cache_response(cache_key, [player.dict() for player in result], ttl=3600)
        
        return result
    except Exception as e:
        logger.error(f"Error searching players: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/players/{player_id}", response_model=PlayerResponse)
async def get_player(player_id: int, db: Session = Depends(get_db)):
    """Get player details by ID"""
    cache_key = get_cache_key("player_detail", player_id=player_id)
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return PlayerResponse(**cached_result)
    
    player = db.query(Player).filter(Player.person_id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    result = PlayerResponse.from_orm(player)
    cache_response(cache_key, result.dict(), ttl=3600)
    
    return result

@app.get("/players/{player_id}/stats", response_model=List[PlayerStatsResponse])
async def get_player_stats(
    player_id: int,
    limit: int = Query(10, ge=1, le=100),
    season: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get player statistics"""
    cache_key = get_cache_key("player_stats", player_id=player_id, limit=limit, season=season)
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return [PlayerStatsResponse(**stat) for stat in cached_result]
    
    query = db.query(PlayerStatistic).filter(PlayerStatistic.person_id == player_id)
    
    if season:
        season_year = int(season.split('-')[0])
        query = query.filter(
            PlayerStatistic.game_date >= datetime(season_year, 10, 1),
            PlayerStatistic.game_date < datetime(season_year + 1, 10, 1)
        )
    
    stats = query.order_by(PlayerStatistic.game_date.desc()).limit(limit).all()
    
    result = [PlayerStatsResponse.from_orm(stat) for stat in stats]
    cache_response(cache_key, [stat.dict() for stat in result], ttl=600)
    
    return result

@app.get("/players/{player_id}/projections", response_model=PlayerProjectionResponse)
async def get_player_projections(
    player_id: int,
    opponent_team_id: Optional[int] = None,
    home_game: bool = True,
    db: Session = Depends(get_db)
):
    """Get player projections for upcoming game"""
    cache_key = get_cache_key(
        "player_projections", 
        player_id=player_id, 
        opponent_team_id=opponent_team_id,
        home_game=home_game
    )
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return PlayerProjectionResponse(**cached_result)
    
    try:
        recent_stats = db.query(PlayerStatistic).filter(
            PlayerStatistic.person_id == player_id
        ).order_by(PlayerStatistic.game_date.desc()).limit(10).all()
        
        if not recent_stats:
            raise HTTPException(status_code=404, detail="No recent stats found for player")
        
        player_features = prepare_player_features(recent_stats, opponent_team_id, home_game)
        
        projections = ml_pipeline.predict_player_stats(player_features)
        
        result = PlayerProjectionResponse(
            player_id=player_id,
            projections=projections,
            confidence_score=0.75,
            last_updated=datetime.utcnow()
        )
        
        cache_response(cache_key, result.dict(), ttl=1800)
        return result
        
    except Exception as e:
        logger.error(f"Error getting player projections: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate projections")

@app.get("/teams", response_model=List[TeamResponse])
async def get_teams(db: Session = Depends(get_db)):
    """Get all teams"""
    cache_key = "all_teams"
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return [TeamResponse(**team) for team in cached_result]
    
    teams = db.query(Team).filter(
        Team.season_active_till.is_(None)
    ).all()
    
    result = [TeamResponse.from_orm(team) for team in teams]
    cache_response(cache_key, [team.dict() for team in result], ttl=7200)
    
    return result

@app.get("/teams/{team_id}", response_model=TeamResponse)
async def get_team(team_id: int, db: Session = Depends(get_db)):
    """Get team details"""
    cache_key = get_cache_key("team_detail", team_id=team_id)
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return TeamResponse(**cached_result)
    
    team = db.query(Team).filter(Team.team_id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    result = TeamResponse.from_orm(team)
    cache_response(cache_key, result.dict(), ttl=3600)
    
    return result

@app.get("/games/today", response_model=List[GameResponse])
async def get_todays_games(db: Session = Depends(get_db)):
    """Get today's games"""
    cache_key = f"todays_games:{datetime.now().date()}"
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return [GameResponse(**game) for game in cached_result]
    
    today = datetime.now().date()
    games = db.query(Game).filter(
        Game.game_date >= today,
        Game.game_date < today + timedelta(days=1)
    ).all()
    
    result = [GameResponse.from_orm(game) for game in games]
    cache_response(cache_key, [game.dict() for game in result], ttl=3600)
    
    return result

@app.post("/predictions/game", response_model=PredictionResponse)
async def predict_game(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Predict game outcome"""
    cache_key = get_cache_key(
        "game_prediction",
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date.date() if request.game_date else None
    )
    
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return PredictionResponse(**cached_result)
    
    try:
        home_features = get_team_features(request.home_team_id, db)
        away_features = get_team_features(request.away_team_id, db)
        
        prediction = ml_pipeline.predict_game_outcome(home_features, away_features)
        
        result = PredictionResponse(
            home_team_id=request.home_team_id,
            away_team_id=request.away_team_id,
            home_win_probability=prediction['home_win_probability'],
            away_win_probability=prediction['away_win_probability'],
            predicted_home_score=prediction['predicted_home_score'],
            predicted_away_score=prediction['predicted_away_score'],
            confidence_score=0.8,
            prediction_date=datetime.utcnow()
        )
        
        cache_response(cache_key, result.dict(), ttl=3600)
        
        background_tasks.add_task(update_team_cache, request.home_team_id, request.away_team_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error predicting game: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate prediction")

@app.get("/analytics/team-strength/{team_id}")
async def get_team_strength(team_id: int, db: Session = Depends(get_db)):
    """Get team strength analytics"""
    cache_key = get_cache_key("team_strength", team_id=team_id)
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return cached_result
    
    # Query analytics tables created by Spark jobs
    try:
        result = db.execute(
            "SELECT * FROM analytics_team_strength WHERE team_id = :team_id",
            {"team_id": team_id}
        ).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Team strength data not found")
        
        analytics_data = dict(result._mapping)
        cache_response(cache_key, analytics_data, ttl=7200)
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error getting team strength: {e}")
        raise HTTPException(status_code=500, detail="Failed to get team analytics")

@app.get("/analytics/player-momentum/{player_id}")
async def get_player_momentum(player_id: int, db: Session = Depends(get_db)):
    """Get player momentum analytics"""
    cache_key = get_cache_key("player_momentum", player_id=player_id)
    cached_result = get_cached_response(cache_key)
    if cached_result:
        return cached_result
    
    try:
        result = db.execute(
            "SELECT * FROM analytics_player_momentum WHERE person_id = :player_id ORDER BY game_date DESC LIMIT 1",
            {"player_id": player_id}
        ).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Player momentum data not found")
        
        momentum_data = dict(result._mapping)
        cache_response(cache_key, momentum_data, ttl=3600)
        
        return momentum_data
        
    except Exception as e:
        logger.error(f"Error getting player momentum: {e}")
        raise HTTPException(status_code=500, detail="Failed to get player analytics")

def prepare_player_features(recent_stats: List[PlayerStatistic], opponent_team_id: Optional[int], home_game: bool) -> Dict:
    """Prepare features for player prediction"""
    if not recent_stats:
        return {}
    
    # Calculate averages from recent games
    avg_points = sum(stat.points for stat in recent_stats) / len(recent_stats)
    avg_assists = sum(stat.assists for stat in recent_stats) / len(recent_stats)
    avg_rebounds = sum(stat.rebounds_total for stat in recent_stats) / len(recent_stats)
    avg_minutes = sum(stat.num_minutes or 0 for stat in recent_stats) / len(recent_stats)
    
    # Recent trend (last 5 vs previous 5)
    if len(recent_stats) >= 10:
        recent_5_avg = sum(stat.points for stat in recent_stats[:5]) / 5
        previous_5_avg = sum(stat.points for stat in recent_stats[5:10]) / 5
        points_trend = recent_5_avg - previous_5_avg
    else:
        points_trend = 0
    
    return {
        'avg_points': avg_points,
        'avg_assists': avg_assists,
        'avg_rebounds': avg_rebounds,
        'avg_minutes': avg_minutes,
        'points_trend': points_trend,
        'home_game': 1 if home_game else 0,
        'rest_days': 2,
        'opponent_strength': 0.5,
    }

def get_team_features(team_id: int, db: Session) -> Dict:
    """Get team features for prediction"""
    # Get recent team performance
    recent_games = db.query(TeamStatistic).filter(
        TeamStatistic.team_id == team_id
    ).order_by(TeamStatistic.game_date.desc()).limit(10).all()
    
    if not recent_games:
        return {}
    
    avg_points_scored = sum(game.team_score for game in recent_games) / len(recent_games)
    avg_points_allowed = sum(game.opponent_score for game in recent_games) / len(recent_games)
    win_rate = sum(1 for game in recent_games if game.win) / len(recent_games)
    avg_fg_pct = sum(game.field_goals_percentage for game in recent_games if game.field_goals_percentage) / len(recent_games)
    
    return {
        'avg_points_scored': avg_points_scored,
        'avg_points_allowed': avg_points_allowed,
        'win_rate': win_rate,
        'avg_fg_percentage': avg_fg_pct,
        'avg_rebounds': sum(game.rebounds_total for game in recent_games) / len(recent_games),
        'avg_assists': sum(game.assists for game in recent_games) / len(recent_games),
        'avg_turnovers': sum(game.turnovers for game in recent_games) / len(recent_games),
    }

async def update_team_cache(home_team_id: int, away_team_id: int):
    """Background task to update team cache"""
    try:
        db = next(get_db())
        
        home_features = get_team_features(home_team_id, db)
        cache_key = get_cache_key("team_features", team_id=home_team_id)
        cache_response(cache_key, home_features, ttl=3600)
        
        away_features = get_team_features(away_team_id, db)
        cache_key = get_cache_key("team_features", team_id=away_team_id)
        cache_response(cache_key, away_features, ttl=3600)
        
        db.close()
        logger.info(f"Updated cache for teams {home_team_id} and {away_team_id}")
        
    except Exception as e:
        logger.error(f"Error updating team cache: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
