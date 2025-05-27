from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Conflict
from config.settings import settings
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_bigquery_client():
    """Get BigQuery client with proper error handling"""
    try:
        return bigquery.Client(project=settings.PROJECT_ID)
    except Exception as e:
        logger.error(f"Failed to create BigQuery client: {e}")
        raise

# Raw Data Schemas
RAW_GAMES_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED", 
                         description="Unique identifier for the game"),
    bigquery.SchemaField("game_date", "DATE", mode="REQUIRED",
                         description="Date the game was played"),
    bigquery.SchemaField("season_id", "STRING", mode="NULLABLE",
                         description="Season identifier, e.g., 22023 for 2023-24 season"),
    bigquery.SchemaField("season_year", "STRING", mode="NULLABLE",
                         description="Year of the season, e.g., 2023-24"),
    bigquery.SchemaField("home_team_id", "INT64", mode="REQUIRED",
                         description="Unique ID for the home team"),
    bigquery.SchemaField("away_team_id", "INT64", mode="REQUIRED", 
                         description="Unique ID for the away team"),
    bigquery.SchemaField("home_team_abbreviation", "STRING", mode="NULLABLE",
                         description="Abbreviation for the home team"),
    bigquery.SchemaField("away_team_abbreviation", "STRING", mode="NULLABLE",
                         description="Abbreviation for the away team"),
    bigquery.SchemaField("home_team_score", "INT64", mode="NULLABLE",
                         description="Final score for the home team"),
    bigquery.SchemaField("away_team_score", "INT64", mode="NULLABLE",
                         description="Final score for the away team"),
    bigquery.SchemaField("wl_home", "STRING", mode="NULLABLE",
                         description="Win (W) or Loss (L) for the home team"),
    bigquery.SchemaField("wl_away", "STRING", mode="NULLABLE",
                         description="Win (W) or Loss (L) for the away team"),
    bigquery.SchemaField("matchup", "STRING", mode="NULLABLE",
                         description="Matchup string, e.g., GSW vs. LAL"),
    bigquery.SchemaField("plus_minus_home", "FLOAT64", mode="NULLABLE",
                         description="Plus/Minus for the home team for the game"),
    bigquery.SchemaField("plus_minus_away", "FLOAT64", mode="NULLABLE",
                         description="Plus/Minus for the away team for the game"),
    bigquery.SchemaField("load_timestamp", "TIMESTAMP", mode="REQUIRED",
                         description="Timestamp of when the record was loaded"),
    bigquery.SchemaField("data_quality_score", "FLOAT64", mode="NULLABLE",
                         description="Data quality score (0-1) for this record")
]

RAW_PLAYER_BOXSCORES_TRADITIONAL_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("player_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("team_abbreviation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("min", "STRING", mode="NULLABLE", 
                         description="Minutes played, e.g., 24:30"),
    bigquery.SchemaField("fgm", "INT64", mode="NULLABLE",
                         description="Field goals made"),
    bigquery.SchemaField("fga", "INT64", mode="NULLABLE",
                         description="Field goals attempted"),
    bigquery.SchemaField("fg_pct", "FLOAT64", mode="NULLABLE",
                         description="Field goal percentage"),
    bigquery.SchemaField("fg3m", "INT64", mode="NULLABLE",
                         description="3-point field goals made"),
    bigquery.SchemaField("fg3a", "INT64", mode="NULLABLE",
                         description="3-point field goals attempted"),
    bigquery.SchemaField("fg3_pct", "FLOAT64", mode="NULLABLE",
                         description="3-point field goal percentage"),
    bigquery.SchemaField("ftm", "INT64", mode="NULLABLE",
                         description="Free throws made"),
    bigquery.SchemaField("fta", "INT64", mode="NULLABLE",
                         description="Free throws attempted"),
    bigquery.SchemaField("ft_pct", "FLOAT64", mode="NULLABLE",
                         description="Free throw percentage"),
    bigquery.SchemaField("oreb", "INT64", mode="NULLABLE",
                         description="Offensive rebounds"),
    bigquery.SchemaField("dreb", "INT64", mode="NULLABLE",
                         description="Defensive rebounds"),
    bigquery.SchemaField("reb", "INT64", mode="NULLABLE",
                         description="Total rebounds"),
    bigquery.SchemaField("ast", "INT64", mode="NULLABLE",
                         description="Assists"),
    bigquery.SchemaField("stl", "INT64", mode="NULLABLE",
                         description="Steals"),
    bigquery.SchemaField("blk", "INT64", mode="NULLABLE",
                         description="Blocks"),
    bigquery.SchemaField("turnover", "INT64", mode="NULLABLE",
                         description="Turnovers (API field is 'TO')"),
    bigquery.SchemaField("pf", "INT64", mode="NULLABLE",
                         description="Personal fouls"),
    bigquery.SchemaField("pts", "INT64", mode="NULLABLE",
                         description="Points scored"),
    bigquery.SchemaField("plus_minus", "FLOAT64", mode="NULLABLE",
                         description="Plus/Minus rating"),
    bigquery.SchemaField("start_position", "STRING", mode="NULLABLE",
                         description="Starting position"),
    bigquery.SchemaField("comment", "STRING", mode="NULLABLE",
                         description="DNP or other comments"),
    bigquery.SchemaField("load_timestamp", "TIMESTAMP", mode="REQUIRED",
                         description="Timestamp of when the record was loaded"),
    bigquery.SchemaField("data_quality_score", "FLOAT64", mode="NULLABLE",
                         description="Data quality score for this player performance")
]

RAW_PLAYER_BOXSCORES_ADVANCED_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("player_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("team_abbreviation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("min", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("e_off_rating", "FLOAT64", mode="NULLABLE",
                         description="Estimated offensive rating"),
    bigquery.SchemaField("off_rating", "FLOAT64", mode="NULLABLE",
                         description="Offensive rating"),
    bigquery.SchemaField("e_def_rating", "FLOAT64", mode="NULLABLE",
                         description="Estimated defensive rating"),
    bigquery.SchemaField("def_rating", "FLOAT64", mode="NULLABLE",
                         description="Defensive rating"),
    bigquery.SchemaField("e_net_rating", "FLOAT64", mode="NULLABLE",
                         description="Estimated net rating"),
    bigquery.SchemaField("net_rating", "FLOAT64", mode="NULLABLE",
                         description="Net rating"),
    bigquery.SchemaField("ast_pct", "FLOAT64", mode="NULLABLE",
                         description="Assist percentage"),
    bigquery.SchemaField("ast_tov", "FLOAT64", mode="NULLABLE",
                         description="Assist to turnover ratio"),
    bigquery.SchemaField("ast_ratio", "FLOAT64", mode="NULLABLE",
                         description="Assist ratio"),
    bigquery.SchemaField("oreb_pct", "FLOAT64", mode="NULLABLE",
                         description="Offensive rebound percentage"),
    bigquery.SchemaField("dreb_pct", "FLOAT64", mode="NULLABLE",
                         description="Defensive rebound percentage"),
    bigquery.SchemaField("reb_pct", "FLOAT64", mode="NULLABLE",
                         description="Total rebound percentage"),
    bigquery.SchemaField("tm_tov_pct", "FLOAT64", mode="NULLABLE",
                         description="Team turnover percentage"),
    bigquery.SchemaField("efg_pct", "FLOAT64", mode="NULLABLE",
                         description="Effective field goal percentage"),
    bigquery.SchemaField("ts_pct", "FLOAT64", mode="NULLABLE",
                         description="True shooting percentage"),
    bigquery.SchemaField("usg_pct", "FLOAT64", mode="NULLABLE",
                         description="Usage percentage"),
    bigquery.SchemaField("e_usg_pct", "FLOAT64", mode="NULLABLE",
                         description="Estimated usage percentage"),
    bigquery.SchemaField("e_pace", "FLOAT64", mode="NULLABLE",
                         description="Estimated pace"),
    bigquery.SchemaField("pace", "FLOAT64", mode="NULLABLE",
                         description="Pace"),
    bigquery.SchemaField("pie", "FLOAT64", mode="NULLABLE",
                         description="Player Impact Estimate"),
    bigquery.SchemaField("start_position", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("comment", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("load_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_quality_score", "FLOAT64", mode="NULLABLE")
]

RAW_TEAM_BOXSCORES_TRADITIONAL_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("team_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("team_abbreviation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("min", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("fgm", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("fga", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("fg_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("fg3m", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("fg3a", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("fg3_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("ftm", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("fta", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("ft_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("oreb", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("dreb", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("reb", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("ast", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("stl", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("blk", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("turnover", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("pf", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("pts", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("plus_minus", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("load_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_quality_score", "FLOAT64", mode="NULLABLE")
]

RAW_TEAM_BOXSCORES_ADVANCED_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("team_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("team_abbreviation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("min", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("e_off_rating", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("off_rating", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("e_def_rating", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("def_rating", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("e_net_rating", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("net_rating", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("ast_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("ast_tov", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("ast_ratio", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("oreb_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("dreb_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("reb_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("e_tm_tov_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("tm_tov_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("efg_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("ts_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("usg_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("e_usg_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("e_pace", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("pace", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("pie", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("load_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_quality_score", "FLOAT64", mode="NULLABLE")
]

RAW_PLAY_BY_PLAY_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("eventnum", "INT64", mode="REQUIRED",
                         description="Event number within the game"),
    bigquery.SchemaField("eventmsgtype", "INT64", mode="NULLABLE",
                         description="Type of event (shot, foul, etc.)"),
    bigquery.SchemaField("eventmsgactiontype", "INT64", mode="NULLABLE",
                         description="Subtype of event"),
    bigquery.SchemaField("period", "INT64", mode="NULLABLE",
                         description="Quarter/period number"),
    bigquery.SchemaField("wctimestring", "STRING", mode="NULLABLE",
                         description="Wall clock time string"),
    bigquery.SchemaField("pctimestring", "STRING", mode="NULLABLE",
                         description="Period clock time string"),
    bigquery.SchemaField("homedescription", "STRING", mode="NULLABLE",
                         description="Home team event description"),
    bigquery.SchemaField("neutraldescription", "STRING", mode="NULLABLE",
                         description="Neutral event description"),
    bigquery.SchemaField("visitordescription", "STRING", mode="NULLABLE",
                         description="Visitor team event description"),
    bigquery.SchemaField("score", "STRING", mode="NULLABLE",
                         description="Score at time of event"),
    bigquery.SchemaField("scoremargin", "STRING", mode="NULLABLE",
                         description="Score margin"),
    bigquery.SchemaField("person1type", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("player1_id", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("player1_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player1_team_id", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("player1_team_city", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player1_team_nickname", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player1_team_abbreviation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("person2type", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("player2_id", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("player2_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player2_team_id", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("player2_team_city", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player2_team_nickname", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player2_team_abbreviation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("person3type", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("player3_id", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("player3_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player3_team_id", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("player3_team_city", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player3_team_nickname", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("player3_team_abbreviation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("video_available_flag", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("load_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("data_quality_score", "FLOAT64", mode="NULLABLE")
]

# Dimension Table Schemas

DIM_PLAYERS_SCHEMA = [
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED",
                         description="PERSON_ID from API"),
    bigquery.SchemaField("full_name", "STRING", mode="NULLABLE",
                         description="DISPLAY_FIRST_LAST from API"),
    bigquery.SchemaField("first_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("last_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("is_active", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("birthdate", "DATE", mode="NULLABLE"),
    bigquery.SchemaField("school", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("country", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("height", "STRING", mode="NULLABLE",
                         description="Height string, e.g., 6-9"),
    bigquery.SchemaField("weight", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("season_exp", "INT64", mode="NULLABLE",
                         description="Seasons of experience"),
    bigquery.SchemaField("jersey", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("position", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("roster_status", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("team_id", "INT64", mode="NULLABLE",
                         description="Current or last known team ID"),
    bigquery.SchemaField("team_abbreviation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("team_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("from_year", "INT64", mode="NULLABLE",
                         description="First season in NBA"),
    bigquery.SchemaField("to_year", "INT64", mode="NULLABLE",
                         description="Last season in NBA"),
    bigquery.SchemaField("draft_year", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("draft_round", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("draft_number", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("load_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("last_updated", "TIMESTAMP", mode="NULLABLE")
]

DIM_TEAMS_SCHEMA = [
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("full_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("abbreviation", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("nickname", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("city", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("state", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("year_founded", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("arena", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("arena_capacity", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("owner", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("general_manager", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("head_coach", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("dleague_affiliation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("load_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("last_updated", "TIMESTAMP", mode="NULLABLE")
]

# Feature Store and Model Output Schemas

FEATURE_STORE_TRAINING_DATA_SCHEMA = [
    # Identifiers
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("game_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("home_team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("away_team_id", "INT64", mode="REQUIRED"),
    
    # Targets
    bigquery.SchemaField("wl_home", "STRING", mode="NULLABLE",
                         description="Win (W) or Loss (L) for home team"),
    bigquery.SchemaField("home_team_score", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("away_team_score", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("point_differential", "INT64", mode="NULLABLE",
                         description="Home score - Away score"),
    
    # Basic Team Features (Last 5 games)
    bigquery.SchemaField("home_team_points_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_team_points_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("home_team_rebounds_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_team_rebounds_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("home_team_assists_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_team_assists_avg_L5", "FLOAT64", mode="NULLABLE"),
    
    # Shooting Features
    bigquery.SchemaField("home_team_fg_pct_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_team_fg_pct_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("home_team_3p_pct_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_team_3p_pct_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("home_team_ts_pct_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_team_ts_pct_avg_L5", "FLOAT64", mode="NULLABLE"),
    
    # Advanced Features
    bigquery.SchemaField("home_offensive_rating_est_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_offensive_rating_est_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("home_avg_def_rating_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_avg_def_rating_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("home_avg_pace_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_avg_pace_avg_L5", "FLOAT64", mode="NULLABLE"),
    
    # Differential Features
    bigquery.SchemaField("diff_team_points_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("diff_offensive_rating_est_avg_L5", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("diff_avg_def_rating_avg_L5", "FLOAT64", mode="NULLABLE"),
    
    # Rest and Schedule Features
    bigquery.SchemaField("home_rest_days", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("away_rest_days", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("rest_advantage", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("home_back_to_back", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("away_back_to_back", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("rest_mismatch", "INT64", mode="NULLABLE"),
    
    # Head-to-Head Features
    bigquery.SchemaField("home_h2h_win_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("home_h2h_avg_margin", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("days_since_last_h2h", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("h2h_games_played", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("margin_volatility", "FLOAT64", mode="NULLABLE"),
    
    # Temporal Features
    bigquery.SchemaField("day_of_week", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("month", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("is_weekend", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("season_progress", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("season_stage", "STRING", mode="NULLABLE"),
    
    # Metadata
    bigquery.SchemaField("features_created_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("feature_engineering_version", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("data_quality_score", "FLOAT64", mode="NULLABLE")
]

PREDICTIONS_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("prediction_date", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("model_version", "STRING", mode="NULLABLE"),
    
    # Win/Loss Predictions
    bigquery.SchemaField("home_win_probability", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("predicted_home_win", "BOOLEAN", mode="NULLABLE"),
    
    # Spread Predictions
    bigquery.SchemaField("predicted_spread", "FLOAT64", mode="REQUIRED",
                         description="Predicted point differential (home - away)"),
    
    # Confidence and Quality Metrics
    bigquery.SchemaField("confidence", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("model_uncertainty", "FLOAT64", mode="NULLABLE"),
    
    # Additional Predictions (if available)
    bigquery.SchemaField("predicted_total_points", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("home_score_prediction", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("away_score_prediction", "FLOAT64", mode="NULLABLE"),
    
    # Feature Importance (top 5 most important features for this prediction)
    bigquery.SchemaField("top_features", "STRING", mode="NULLABLE",
                         description="JSON string of top feature importances"),
    
    # Metadata
    bigquery.SchemaField("features_used_count", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("prediction_runtime_ms", "FLOAT64", mode="NULLABLE")
]

MODEL_TRAINING_HISTORY_SCHEMA = [
    bigquery.SchemaField("training_date", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("model_version", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("feature_engineering_version", "STRING", mode="NULLABLE"),
    
    # Dataset Information
    bigquery.SchemaField("training_samples", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("validation_samples", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("test_samples", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("feature_count", "INT64", mode="NULLABLE"),
    
    # Win/Loss Model Performance
    bigquery.SchemaField("win_loss_accuracy", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("win_loss_precision", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("win_loss_recall", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("win_loss_f1", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("win_loss_auc", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("win_loss_log_loss", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("win_loss_brier_score", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("win_loss_expected_calibration_error", "FLOAT64", mode="NULLABLE"),
    
    # Spread Model Performance
    bigquery.SchemaField("spread_mae", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("spread_mse", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("spread_rmse", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("spread_mean_target", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("spread_std_target", "FLOAT64", mode="NULLABLE"),
    
    # Training Configuration
    bigquery.SchemaField("learning_rate", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("batch_size", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("epochs_trained", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("early_stopping_epoch", "INT64", mode="NULLABLE"),
    
    # Model Storage
    bigquery.SchemaField("model_path", "STRING", mode="NULLABLE",
                         description="GCS path to saved model"),
    bigquery.SchemaField("model_size_mb", "FLOAT64", mode="NULLABLE"),
    
    # Training Performance
    bigquery.SchemaField("training_duration_minutes", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("gpu_used", "BOOLEAN", mode="NULLABLE"),
    
    # Data Quality at Training Time
    bigquery.SchemaField("avg_data_quality_score", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("min_data_quality_score", "FLOAT64", mode="NULLABLE")
]

BETTING_ODDS_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("book_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("market_type", "STRING", mode="REQUIRED",
                         description="moneyline, spread, total, etc."),
    bigquery.SchemaField("team_type", "STRING", mode="REQUIRED",
                         description="home, away, over, under"),
    
    # Odds Information
    bigquery.SchemaField("odds", "FLOAT64", mode="NULLABLE",
                         description="Current odds (American format)"),
    bigquery.SchemaField("decimal_odds", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("implied_probability", "FLOAT64", mode="NULLABLE"),
    
    # Spread/Total Information
    bigquery.SchemaField("spread", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("total", "FLOAT64", mode="NULLABLE"),
    
    # Historical Tracking
    bigquery.SchemaField("opening_odds", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("opening_spread", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("opening_total", "FLOAT64", mode="NULLABLE"),
    
    # Movement Tracking
    bigquery.SchemaField("odds_movement", "FLOAT64", mode="NULLABLE",
                         description="Current odds - Opening odds"),
    bigquery.SchemaField("spread_movement", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("total_movement", "FLOAT64", mode="NULLABLE"),
    
    # Metadata
    bigquery.SchemaField("collected_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("odds_last_updated", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("country_code", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("data_source", "STRING", mode="NULLABLE")
]

# Backwards compatibility with existing code
GAME_STATS_SCHEMA = RAW_GAMES_SCHEMA
PLAYER_STATS_SCHEMA = RAW_PLAYER_BOXSCORES_TRADITIONAL_SCHEMA

# Table Configuration with Cost Optimization
TABLE_CONFIGURATIONS = {
    "raw_games": {
        "schema": RAW_GAMES_SCHEMA,
        "partition_field": "game_date",
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["home_team_id", "away_team_id"],
        "partition_expiration_days": settings.PARTITION_EXPIRATION_DAYS,
        "description": "Raw game-level information including scores and matchups, partitioned by game_date and clustered by team IDs."
    },
    "raw_player_boxscores_traditional": {
        "schema": RAW_PLAYER_BOXSCORES_TRADITIONAL_SCHEMA,
        "partition_field": "_PARTITIONDATE",  # Will use DATE(load_timestamp)
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["player_id", "team_id", "game_id"],
        "partition_expiration_days": settings.PARTITION_EXPIRATION_DAYS,
        "description": "Raw traditional player box scores, partitioned by load date and clustered for optimal query performance."
    },
    "raw_player_boxscores_advanced": {
        "schema": RAW_PLAYER_BOXSCORES_ADVANCED_SCHEMA,
        "partition_field": "_PARTITIONDATE",
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["player_id", "team_id", "game_id"],
        "partition_expiration_days": settings.PARTITION_EXPIRATION_DAYS,
        "description": "Raw advanced player box scores with advanced metrics."
    },
    "raw_team_boxscores_traditional": {
        "schema": RAW_TEAM_BOXSCORES_TRADITIONAL_SCHEMA,
        "partition_field": "_PARTITIONDATE",
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["team_id", "game_id"],
        "partition_expiration_days": settings.PARTITION_EXPIRATION_DAYS,
        "description": "Raw traditional team box scores aggregated at team level."
    },
    "raw_team_boxscores_advanced": {
        "schema": RAW_TEAM_BOXSCORES_ADVANCED_SCHEMA,
        "partition_field": "_PARTITIONDATE",
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["team_id", "game_id"],
        "partition_expiration_days": settings.PARTITION_EXPIRATION_DAYS,
        "description": "Raw advanced team box scores with efficiency metrics."
    },
    "raw_play_by_play": {
        "schema": RAW_PLAY_BY_PLAY_SCHEMA,
        "partition_field": "_PARTITIONDATE",
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["game_id", "period", "eventnum"],
        "partition_expiration_days": settings.PBP_PARTITION_EXPIRATION_DAYS,  # Shorter retention
        "description": "Raw play-by-play events, partitioned by load date and clustered for chronological access."
    },
    "dim_players": {
        "schema": DIM_PLAYERS_SCHEMA,
        "partition_field": None,  # Dimension tables typically not partitioned
        "clustering_fields": ["player_id"],
        "description": "Player dimension table with biographical and career information."
    },
    "dim_teams": {
        "schema": DIM_TEAMS_SCHEMA,
        "partition_field": None,
        "clustering_fields": ["team_id"],
        "description": "Team dimension table with franchise information."
    },
    "feature_set": {
        "schema": FEATURE_STORE_TRAINING_DATA_SCHEMA,
        "partition_field": "game_date",
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["game_id", "home_team_id", "away_team_id"],
        "partition_expiration_days": settings.PARTITION_EXPIRATION_DAYS * 2,  # Keep features longer
        "description": "Engineered features for model training, partitioned by game date."
    },
    "predictions": {
        "schema": PREDICTIONS_SCHEMA,
        "partition_field": "prediction_date",
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["game_id"],
        "partition_expiration_days": 365,  # Keep predictions for 1 year
        "description": "Model predictions with confidence metrics, partitioned by prediction date."
    },
    "model_training_history": {
        "schema": MODEL_TRAINING_HISTORY_SCHEMA,
        "partition_field": "training_date",
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["model_version"],
        "description": "Historical record of model training runs and performance metrics."
    },
    "betting_odds": {
        "schema": BETTING_ODDS_SCHEMA,
        "partition_field": "collected_at",
        "partition_type": bigquery.TimePartitioningType.DAY,
        "clustering_fields": ["game_id", "book_name", "market_type"],
        "partition_expiration_days": 90,  # Keep odds for 3 months
        "description": "Betting odds from various sportsbooks for analysis and value identification."
    }
}

def create_dataset_if_not_exists():
    """Create dataset with proper configuration and error handling"""
    try:
        client = get_bigquery_client()
        dataset_id = f"{settings.PROJECT_ID}.{settings.DATASET_ID}"
        
        try:
            dataset = client.get_dataset(dataset_id)
            logger.info(f"Dataset {dataset.dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = settings.BIGQUERY_LOCATION
            dataset.description = f"NBA betting data warehouse created on {datetime.now().isoformat()}"
            
            # Set dataset labels for cost tracking
            dataset.labels = {
                "project": "nba-betting",
                "environment": "production",
                "cost-center": "analytics"
            }
            
            dataset = client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset.dataset_id} in {dataset.location}")
        
        return client
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise

def create_table_with_optimization(client: bigquery.Client, table_name: str, 
                                 config: Dict) -> bigquery.Table:
    """Create table with cost-optimized configuration"""
    try:
        dataset_ref = client.dataset(settings.DATASET_ID)
        table_ref = dataset_ref.table(table_name)
        
        # Check if table already exists
        try:
            existing_table = client.get_table(table_ref)
            logger.info(f"Table {table_name} already exists with {existing_table.num_rows} rows")
            return existing_table
        except NotFound:
            pass
        
        # Create new table
        table = bigquery.Table(table_ref, schema=config["schema"])
        table.description = config.get("description", f"Table {table_name}")
        
        # Set partitioning if specified
        if config.get("partition_field"):
            if config["partition_field"] == "_PARTITIONDATE":
                # Ingestion-time partitioning
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=config.get("partition_type", bigquery.TimePartitioningType.DAY)
                )
            else:
                # Column-based partitioning
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=config.get("partition_type", bigquery.TimePartitioningType.DAY),
                    field=config["partition_field"]
                )
            
            # Set partition expiration if specified
            if config.get("partition_expiration_days"):
                expiration_ms = config["partition_expiration_days"] * 24 * 60 * 60 * 1000
                table.time_partitioning.expiration_ms = expiration_ms
                logger.info(f"Set partition expiration: {config['partition_expiration_days']} days")
        
        # Set clustering if specified
        if config.get("clustering_fields"):
            table.clustering_fields = config["clustering_fields"]
            logger.info(f"Set clustering fields: {config['clustering_fields']}")
        
        # Set table labels for cost tracking
        table.labels = {
            "table-type": "raw-data" if table_name.startswith("raw_") else "processed-data",
            "partitioned": "true" if config.get("partition_field") else "false",
            "clustered": "true" if config.get("clustering_fields") else "false"
        }
        
        # Create the table
        table = client.create_table(table, timeout=30)
        logger.info(f"Created optimized table {table_name}")
        
        return table
        
    except Exception as e:
        logger.error(f"Error creating table {table_name}: {e}")
        raise

def create_optimized_tables():
    """Create all tables with cost optimization features"""
    try:
        client = create_dataset_if_not_exists()
        
        created_tables = []
        failed_tables = []
        
        for table_name, config in TABLE_CONFIGURATIONS.items():
            try:
                table = create_table_with_optimization(client, table_name, config)
                created_tables.append(table_name)
                logger.info(f"✓ Table {table_name} ready")
            except Exception as e:
                logger.error(f"✗ Failed to create table {table_name}: {e}")
                failed_tables.append((table_name, str(e)))
        
        # Summary
        logger.info(f"Table creation summary:")
        logger.info(f"  Successfully created/verified: {len(created_tables)}")
        logger.info(f"  Failed: {len(failed_tables)}")
        
        if failed_tables:
            for table_name, error in failed_tables:
                logger.error(f"    {table_name}: {error}")
            raise Exception(f"Failed to create {len(failed_tables)} tables")
        
        logger.info("All tables created successfully with cost optimizations!")
        return client
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def validate_table_schema(client: bigquery.Client, table_name: str, 
                         expected_schema: List[bigquery.SchemaField]) -> bool:
    """Validate that table schema matches expected schema"""
    try:
        table_ref = client.dataset(settings.DATASET_ID).table(table_name)
        table = client.get_table(table_ref)
        
        actual_fields = {field.name: field for field in table.schema}
        expected_fields = {field.name: field for field in expected_schema}
        
        # Check for missing fields
        missing_fields = set(expected_fields.keys()) - set(actual_fields.keys())
        extra_fields = set(actual_fields.keys()) - set(expected_fields.keys())
        
        if missing_fields:
            logger.warning(f"Table {table_name} missing fields: {missing_fields}")
            return False
        
        if extra_fields:
            logger.info(f"Table {table_name} has extra fields: {extra_fields}")
        
        # Check field types for existing fields
        type_mismatches = []
        for field_name in expected_fields:
            if field_name in actual_fields:
                expected_type = expected_fields[field_name].field_type
                actual_type = actual_fields[field_name].field_type
                if expected_type != actual_type:
                    type_mismatches.append(f"{field_name}: expected {expected_type}, got {actual_type}")
        
        if type_mismatches:
            logger.warning(f"Table {table_name} type mismatches: {type_mismatches}")
            return False
        
        logger.info(f"Table {table_name} schema validation passed")
        return True
        
    except NotFound:
        logger.error(f"Table {table_name} does not exist")
        return False
    except Exception as e:
        logger.error(f"Error validating schema for {table_name}: {e}")
        return False

def get_table_cost_info(client: bigquery.Client, table_name: str) -> Dict:
    """Get cost-related information about a table"""
    try:
        table_ref = client.dataset(settings.DATASET_ID).table(table_name)
        table = client.get_table(table_ref)
        
        info = {
            "table_name": table_name,
            "num_rows": table.num_rows,
            "num_bytes": table.num_bytes,
            "size_mb": round(table.num_bytes / 1024 / 1024, 2),
            "partitioned": table.time_partitioning is not None,
            "clustered": table.clustering_fields is not None,
            "created": table.created.isoformat() if table.created else None,
            "modified": table.modified.isoformat() if table.modified else None
        }
        
        if table.time_partitioning:
            info["partition_field"] = table.time_partitioning.field
            info["partition_type"] = table.time_partitioning.type_
            if table.time_partitioning.expiration_ms:
                info["partition_expiration_days"] = table.time_partitioning.expiration_ms // (24 * 60 * 60 * 1000)
        
        if table.clustering_fields:
            info["clustering_fields"] = table.clustering_fields
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting cost info for {table_name}: {e}")
        return {"table_name": table_name, "error": str(e)}

def generate_cost_optimization_report(client: bigquery.Client = None) -> Dict:
    """Generate comprehensive cost optimization report"""
    if client is None:
        client = get_bigquery_client()
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset": settings.DATASET_ID,
        "tables": {},
        "summary": {
            "total_tables": 0,
            "total_size_mb": 0,
            "partitioned_tables": 0,
            "clustered_tables": 0,
            "tables_with_expiration": 0
        }
    }
    
    for table_name in TABLE_CONFIGURATIONS.keys():
        try:
            table_info = get_table_cost_info(client, table_name)
            report["tables"][table_name] = table_info
            
            # Update summary
            report["summary"]["total_tables"] += 1
            if table_info.get("size_mb"):
                report["summary"]["total_size_mb"] += table_info["size_mb"]
            if table_info.get("partitioned"):
                report["summary"]["partitioned_tables"] += 1
            if table_info.get("clustered"):
                report["summary"]["clustered_tables"] += 1
            if table_info.get("partition_expiration_days"):
                report["summary"]["tables_with_expiration"] += 1
                
        except Exception as e:
            logger.warning(f"Could not get info for table {table_name}: {e}")
    
    # Calculate optimization percentage
    total_tables = report["summary"]["total_tables"]
    if total_tables > 0:
        report["summary"]["partitioning_coverage"] = round(
            report["summary"]["partitioned_tables"] / total_tables * 100, 1
        )
        report["summary"]["clustering_coverage"] = round(
            report["summary"]["clustered_tables"] / total_tables * 100, 1
        )
    
    return report

def cleanup_expired_partitions(client: bigquery.Client = None, dry_run: bool = True):
    """Manually cleanup expired partitions (for tables without automatic expiration)"""
    if client is None:
        client = get_bigquery_client()
    
    cleanup_results = []
    
    for table_name, config in TABLE_CONFIGURATIONS.items():
        if not config.get("partition_expiration_days"):
            continue
            
        try:
            expiration_days = config["partition_expiration_days"]
            cutoff_date = (datetime.now() - timedelta(days=expiration_days)).date()
            
            # Query to find old partitions
            query = f"""
            SELECT 
                partition_id,
                creation_time,
                last_modified_time,
                total_rows,
                total_logical_bytes
            FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.INFORMATION_SCHEMA.PARTITIONS`
            WHERE table_name = '{table_name}'
            AND partition_id IS NOT NULL
            AND partition_id != '__NULL__'
            AND PARSE_DATE('%Y%m%d', partition_id) < '{cutoff_date}'
            ORDER BY partition_id
            """
            
            old_partitions = client.query(query).to_dataframe()
            
            if not old_partitions.empty:
                logger.info(f"Found {len(old_partitions)} old partitions in {table_name}")
                
                if not dry_run:
                    # Delete old partitions (implementation would go here)
                    # This is complex and should be done carefully
                    logger.warning("Partition deletion not implemented - use automatic expiration instead")
                
                cleanup_results.append({
                    "table": table_name,
                    "old_partitions": len(old_partitions),
                    "bytes_to_cleanup": old_partitions["total_logical_bytes"].sum()
                })
                
        except Exception as e:
            logger.error(f"Error checking partitions for {table_name}: {e}")
    
    return cleanup_results

# Convenience functions for backwards compatibility
def initialize_database():
    """Initialize all required tables (legacy function name)"""
    return create_optimized_tables()

if __name__ == "__main__":
    # CLI for database management
    import argparse
    
    parser = argparse.ArgumentParser(description="NBA BigQuery Schema Management")
    parser.add_argument("--action", choices=["create", "validate", "report", "cleanup"], 
                       default="create", help="Action to perform")
    parser.add_argument("--table", help="Specific table to operate on")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    if args.action == "create":
        create_optimized_tables()
    elif args.action == "validate":
        client = get_bigquery_client()
        if args.table:
            if args.table in TABLE_CONFIGURATIONS:
                validate_table_schema(client, args.table, TABLE_CONFIGURATIONS[args.table]["schema"])
            else:
                logger.error(f"Unknown table: {args.table}")
        else:
            for table_name, config in TABLE_CONFIGURATIONS.items():
                validate_table_schema(client, table_name, config["schema"])
    elif args.action == "report":
        report = generate_cost_optimization_report()
        print(f"Cost Optimization Report for {report['dataset']}:")
        print(f"  Total tables: {report['summary']['total_tables']}")
        print(f"  Total size: {report['summary']['total_size_mb']:.1f} MB")
        print(f"  Partitioning coverage: {report['summary'].get('partitioning_coverage', 0)}%")
        print(f"  Clustering coverage: {report['summary'].get('clustering_coverage', 0)}%")
    elif args.action == "cleanup":
        cleanup_results = cleanup_expired_partitions(dry_run=args.dry_run)
        print(f"Cleanup results: {cleanup_results}")