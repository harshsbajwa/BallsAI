from google.cloud import bigquery
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

def get_bigquery_client():
    """Get BigQuery client with proper error handling"""
    try:
        return bigquery.Client(project=settings.PROJECT_ID)
    except Exception as e:
        logger.error(f"Failed to create BigQuery client: {e}")
        raise

# Schema definitions
GAME_STATS_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("game_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("season", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("home_team_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("away_team_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("home_team_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("away_team_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("home_score", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("away_score", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("home_win", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("game_status", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("game_status_id", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")
]

BETTING_ODDS_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("book_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("market_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("team_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("odds", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("spread", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("opening_odds", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("opening_spread", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("odds_trend", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("collected_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("country_code", "STRING", mode="NULLABLE")
]

PLAYER_STATS_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("player_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("team_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("player_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("position", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("minutes", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("points", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("rebounds", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("assists", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("steals", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("blocks", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("turnovers", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("field_goals_made", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("field_goals_attempted", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("three_pointers_made", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("three_pointers_attempted", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("free_throws_made", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("free_throws_attempted", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("plus_minus", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")
]

FEATURE_SET_SCHEMA = [
    # Game info
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("game_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("home_team_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("away_team_id", "INTEGER", mode="REQUIRED"),
    
    # Targets
    bigquery.SchemaField("home_win", "BOOLEAN", mode="NULLABLE"),
    bigquery.SchemaField("home_score", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("away_score", "INTEGER", mode="NULLABLE"),
    
    # Team features
    bigquery.SchemaField("home_team_points_avg_5", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("away_team_points_avg_5", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("home_rest_days", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("away_rest_days", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("rest_advantage", "INTEGER", mode="NULLABLE"),
    
    # Time features
    bigquery.SchemaField("day_of_week", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("month", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("is_weekend", "INTEGER", mode="NULLABLE"),
    
    # Metadata
    bigquery.SchemaField("features_created_at", "TIMESTAMP", mode="REQUIRED")
]

PREDICTIONS_SCHEMA = [
    bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("prediction_date", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("home_win_probability", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("predicted_spread", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("confidence", "FLOAT", mode="NULLABLE")
]

MODEL_TRAINING_HISTORY_SCHEMA = [
    bigquery.SchemaField("training_date", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("training_samples", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("test_samples", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("win_loss_accuracy", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("win_loss_auc", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("spread_mae", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("spread_rmse", "FLOAT", mode="NULLABLE")
]


def create_dataset_if_not_exists():
    """Create dataset with proper error handling"""
    try:
        client = get_bigquery_client()
        dataset_id = f"{settings.PROJECT_ID}.{settings.DATASET_ID}"
        
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        
        dataset = client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset {dataset.dataset_id} ready")
        return client
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise


def create_table_if_not_exists(client, table_name, schema):
    """Create table with better error handling"""
    try:
        dataset_ref = client.dataset(settings.DATASET_ID)
        table_ref = dataset_ref.table(table_name)
        
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table, exists_ok=True)
        logger.info(f"Table {table_name} ready")
        return table
    except Exception as e:
        logger.error(f"Error creating table {table_name}: {e}")
        raise


def initialize_database():
    """Initialize all required tables"""
    try:
        client = create_dataset_if_not_exists()
        
        tables_to_create = [
            ("game_stats", GAME_STATS_SCHEMA),
            ("betting_odds", BETTING_ODDS_SCHEMA),
            ("player_stats", PLAYER_STATS_SCHEMA),
            ("feature_set", FEATURE_SET_SCHEMA),
            ("predictions", PREDICTIONS_SCHEMA),
            ("model_training_history", MODEL_TRAINING_HISTORY_SCHEMA)
        ]
        
        for table_name, schema in tables_to_create:
            create_table_if_not_exists(client, table_name, schema)
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


if __name__ == "__main__":
    initialize_database()