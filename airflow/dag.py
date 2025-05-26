from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.operators.bigquery import *
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import logging

from data.collector import NBADataCollector
from data.spark import NBAFeatureEngineer
from notebooks.models import NBAModelTrainer, NBAPredictor
from google.cloud import bigquery
from config.settings import settings

logger = logging.getLogger(__name__)

# DAG Configuration
default_args = {
    'owner': 'nba-betting-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['admin@company.com'],  # Add your email
}

dag = DAG(
    'nba_betting_ml_pipeline',
    default_args=default_args,
    description='Production NBA betting ML pipeline with BigQuery operators',
    schedule_interval='0 10 * * *',  # Daily at 10 AM
    catchup=False,
    max_active_runs=1,
    tags=['nba', 'betting', 'ml', 'production'],
)

# Constants
PROJECT_ID = settings.PROJECT_ID
DATASET_ID = settings.DATASET_ID
GCS_BUCKET = f"{PROJECT_ID}-nba-data"
LOCATION = "US"

# SQL Queries
CREATE_FEATURE_SET_QUERY = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.feature_set` AS
WITH team_stats AS (
    SELECT
        ps.game_id,
        ps.team_id,
        AVG(ps.points) as avg_points,
        AVG(ps.rebounds) as avg_rebounds,
        AVG(ps.assists) as avg_assists,
        AVG(CASE WHEN ps.field_goals_attempted > 0
            THEN ps.field_goals_made / ps.field_goals_attempted
            ELSE 0 END) as avg_fg_pct,
        AVG(CASE WHEN ps.three_pointers_attempted > 0
            THEN ps.three_pointers_made / ps.three_pointers_attempted
            ELSE 0 END) as avg_3p_pct,
        SUM(ps.points) as team_points,
        SUM(ps.rebounds) as team_rebounds,
        SUM(ps.assists) as team_assists,
        SUM(ps.turnovers) as team_turnovers
    FROM `{PROJECT_ID}.{DATASET_ID}.player_stats` ps
    GROUP BY ps.game_id, ps.team_id
),
home_stats AS (
    SELECT
        g.game_id,
        g.game_date,
        g.home_team_id,
        g.away_team_id,
        g.home_win,
        g.home_score,
        g.away_score,
        ts.avg_points as home_avg_points,
        ts.avg_rebounds as home_avg_rebounds,
        ts.avg_assists as home_avg_assists,
        ts.avg_fg_pct as home_avg_fg_pct,
        ts.avg_3p_pct as home_avg_3p_pct,
        ts.team_points as home_team_points,
        ts.team_rebounds as home_team_rebounds,
        ts.team_assists as home_team_assists,
        ts.team_turnovers as home_team_turnovers
    FROM `{PROJECT_ID}.{DATASET_ID}.game_stats` g
    LEFT JOIN team_stats ts ON g.game_id = ts.game_id AND g.home_team_id = ts.team_id
),
away_stats AS (
    SELECT
        game_id,
        ts.avg_points as away_avg_points,
        ts.avg_rebounds as away_avg_rebounds,
        ts.avg_assists as away_avg_assists,
        ts.avg_fg_pct as away_avg_fg_pct,
        ts.avg_3p_pct as away_avg_3p_pct,
        ts.team_points as away_team_points,
        ts.team_rebounds as away_team_rebounds,
        ts.team_assists as away_team_assists,
        ts.team_turnovers as away_team_turnovers
    FROM team_stats ts
)
SELECT
    hs.*,
    aws.away_avg_points,
    aws.away_avg_rebounds,
    aws.away_avg_assists,
    aws.away_avg_fg_pct,
    aws.away_avg_3p_pct,
    aws.away_team_points,
    aws.away_team_rebounds,
    aws.away_team_assists,
    aws.away_team_turnovers,
    EXTRACT(DAYOFWEEK FROM hs.game_date) as day_of_week,
    EXTRACT(MONTH FROM hs.game_date) as month,
    CASE WHEN EXTRACT(DAYOFWEEK FROM hs.game_date) IN (1, 7) THEN 1 ELSE 0 END as is_weekend,
    (hs.home_avg_points - aws.away_avg_points) as points_differential,
    (hs.home_avg_rebounds - aws.away_avg_rebounds) as rebounds_differential,
    (hs.home_avg_assists - aws.away_avg_assists) as assists_differential,
    CURRENT_TIMESTAMP() as features_created_at
FROM home_stats hs
LEFT JOIN away_stats aws ON hs.game_id = aws.game_id
WHERE hs.game_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY)
"""

DATA_QUALITY_CHECK_QUERY = f"""
SELECT 
    COUNT(*) as total_records,
    COUNT(CASE WHEN home_win IS NULL THEN 1 END) as null_targets,
    COUNT(CASE WHEN home_avg_points IS NULL THEN 1 END) as null_features,
    MIN(game_date) as earliest_date,
    MAX(game_date) as latest_date
FROM `{PROJECT_ID}.{DATASET_ID}.feature_set`
"""

def safe_task_wrapper(func):
    """Wrapper for safe task execution"""
    def wrapper(**context):
        try:
            return func(**context)
        except Exception as e:
            logger.error(f"Task {func.__name__} failed: {e}")
            # Send notification or alert here
            raise
    return wrapper


@safe_task_wrapper
def collect_daily_data(**context):
    """Collect daily NBA games and player stats"""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        collector = NBADataCollector(client)

        # Collect games
        games = collector.collect_daily_games()
        if games:
            collector.upload_to_bigquery("game_stats", games)
            logger.info(f"Uploaded {len(games)} games")

        # Collect completed game stats
        completed_games = [g['game_id'] for g in games if g.get('game_status_id') == 3]
        if completed_games:
            player_stats = collector.collect_box_scores(completed_games)
            if player_stats:
                collector.upload_to_bigquery("player_stats", player_stats)
                logger.info(f"Uploaded {len(player_stats)} player stat records")

        # Try to collect betting odds
        try:
            odds = collector.collect_betting_odds()
            if odds:
                collector.upload_to_bigquery("betting_odds", odds)
                logger.info(f"Uploaded {len(odds)} betting odds")
        except Exception as e:
            logger.warning(f"Could not collect betting odds: {e}")

        return {"games_collected": len(games), "completed_games": len(completed_games)}

    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        raise


def should_retrain_model(**context):
    """Determine if model should be retrained"""
    try:
        should_retrain = check_model_performance(**context)
        if should_retrain:
            return 'train_models'
        else:
            return 'make_daily_predictions'
    except Exception as e:
        logger.error(f"Error checking model performance: {e}")
        return 'make_daily_predictions'  # Default to predictions


def train_models(**context):
    """Train ML models"""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        trainer = NBAModelTrainer(client)
        model_data = trainer.train_all_models()
        
        # Store training metrics
        training_metrics = [{
            'training_date': model_data['training_date'],
            'training_samples': model_data['training_samples'],
            'test_samples': model_data['test_samples'],
            'win_loss_accuracy': model_data['model_performance']['win_loss']['accuracy'],
            'win_loss_auc': model_data['model_performance']['win_loss']['auc'],
            'spread_mae': model_data['model_performance']['spread']['mae'],
            'spread_rmse': model_data['model_performance']['spread']['rmse']
        }]
        
        client.load_table_from_json(
            training_metrics, 
            f"{PROJECT_ID}.{DATASET_ID}.model_training_history"
        ).result()
        
        logger.info("Model training completed successfully")
        return model_data['model_performance']
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

def make_daily_predictions(**context):
    """Make predictions for upcoming games"""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        predictor = NBAPredictor()
        
        # Get upcoming games
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.feature_set`
        WHERE game_date = CURRENT_DATE()
        AND home_win IS NULL
        """
        
        upcoming_games = client.query(query).to_dataframe()
        
        predictions = []
        for _, game in upcoming_games.iterrows():
            features = game.to_dict()
            prediction = predictor.predict_game(features)
            
            if prediction:
                pred_record = {
                    'game_id': game['game_id'],
                    'prediction_date': datetime.now().isoformat(),
                    'home_win_probability': prediction['home_win_probability'],
                    'predicted_spread': prediction['predicted_spread'],
                    'confidence': prediction['confidence']
                }
                predictions.append(pred_record)
        
        if predictions:
            client.load_table_from_json(
                predictions,
                f"{PROJECT_ID}.{DATASET_ID}.predictions"
            ).result()
            logger.info(f"Made predictions for {len(predictions)} games")
        
        return {"predictions_made": len(predictions)}
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def check_model_performance(**context):
    """Check if model retraining is needed"""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        
        # Check recent performance
        query = f"""
        SELECT AVG(correct_prediction) as recent_accuracy
        FROM (
            SELECT 
                CASE 
                    WHEN p.home_win_probability > 0.5 AND g.home_win THEN 1
                    WHEN p.home_win_probability <= 0.5 AND NOT g.home_win THEN 1
                    ELSE 0
                END as correct_prediction
            FROM `{PROJECT_ID}.{DATASET_ID}.predictions` p
            JOIN `{PROJECT_ID}.{DATASET_ID}.game_stats` g ON p.game_id = g.game_id
            WHERE p.prediction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
            AND g.home_win IS NOT NULL
        )
        """
        
        result = client.query(query).to_dataframe()
        
        if not result.empty and result.iloc[0]['recent_accuracy'] is not None:
            recent_accuracy = result.iloc[0]['recent_accuracy']
            logger.info(f"Recent model accuracy: {recent_accuracy:.3f}")
            
            # Trigger retraining if accuracy drops below threshold
            if recent_accuracy < 0.52:  # Below 52% accuracy
                logger.warning("Model performance degraded, triggering retraining")
                return True
        
        # Also retrain weekly
        return datetime.now().weekday() == 0  # Monday
        
    except Exception as e:
        logger.error(f"Error checking model performance: {e}")
        return False

# Task Definitions

# 1. Infrastructure Setup
create_dataset = BigQueryCreateEmptyDatasetOperator(
    task_id="create_dataset",
    dataset_id=DATASET_ID,
    project_id=PROJECT_ID,
    location=LOCATION,
    dag=dag,
)

# 2. Data Collection
collect_data_task = PythonOperator(
    task_id='collect_daily_data',
    python_callable=collect_daily_data,
    retries=3,
    retry_delay=timedelta(minutes=10),
    dag=dag,
)

# 3. Data Quality Checks
check_raw_data = BigQueryCheckOperator(
    task_id="check_raw_data_exists",
    sql=f"SELECT COUNT(*) FROM `{PROJECT_ID}.{DATASET_ID}.game_stats` WHERE game_date = CURRENT_DATE()",
    use_legacy_sql=False,
    location=LOCATION,
    dag=dag,
)

# 4. Feature Engineering
create_features = BigQueryInsertJobOperator(
    task_id="create_features",
    configuration={
        "query": {
            "query": CREATE_FEATURE_SET_QUERY,
            "useLegacySql": False,
            "writeDisposition": "WRITE_TRUNCATE",
        }
    },
    location=LOCATION,
    dag=dag,
)

# 5. Feature Quality Validation
validate_features = BigQueryValueCheckOperator(
    task_id="validate_feature_quality",
    sql=f"SELECT COUNT(*) FROM `{PROJECT_ID}.{DATASET_ID}.feature_set` WHERE features_created_at >= CURRENT_DATE()",
    pass_value=0,
    tolerance=None,
    use_legacy_sql=False,
    location=LOCATION,
    dag=dag,
)

# 6. Data Quality Report
generate_quality_report = BigQueryInsertJobOperator(
    task_id="generate_quality_report",
    configuration={
        "query": {
            "query": DATA_QUALITY_CHECK_QUERY,
            "useLegacySql": False,
        }
    },
    location=LOCATION,
    dag=dag,
)

# 7. Model Performance Check
check_performance = PythonOperator(
    task_id='check_model_performance',
    python_callable=check_model_performance,
    dag=dag,
)

# 8. Conditional training branch
training_branch = BranchPythonOperator(
    task_id='check_training_needed',
    python_callable=should_retrain_model,
    dag=dag,
)

# 9. Model Training
train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    retries=1,
    retry_delay=timedelta(minutes=30),
    execution_timeout=timedelta(hours=2),
    dag=dag,
)

# 10. Daily Predictions
make_predictions_task = PythonOperator(
    task_id='make_daily_predictions',
    python_callable=make_daily_predictions,
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

# 11. Model Evaluation
evaluate_predictions = BigQueryInsertJobOperator(
    task_id="evaluate_recent_predictions",
    configuration={
        "query": {
            "query": f"""
            CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.daily_evaluation` AS
            SELECT 
                CURRENT_DATE() as evaluation_date,
                COUNT(*) as total_predictions,
                AVG(CASE 
                    WHEN p.home_win_probability > 0.5 AND g.home_win THEN 1
                    WHEN p.home_win_probability <= 0.5 AND NOT g.home_win THEN 1
                    ELSE 0
                END) as accuracy,
                AVG(ABS(p.predicted_spread - (g.home_score - g.away_score))) as avg_spread_error
            FROM `{PROJECT_ID}.{DATASET_ID}.predictions` p
            JOIN `{PROJECT_ID}.{DATASET_ID}.game_stats` g ON p.game_id = g.game_id
            WHERE p.prediction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
            AND g.home_win IS NOT NULL
            """,
            "useLegacySql": False,
        }
    },
    location=LOCATION,
    dag=dag,
)

# 12. Cleanup Old Data
cleanup_old_predictions = BigQueryInsertJobOperator(
    task_id="cleanup_old_predictions",
    configuration={
        "query": {
            "query": f"""
            DELETE FROM `{PROJECT_ID}.{DATASET_ID}.predictions`
            WHERE prediction_date < DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
            """,
            "useLegacySql": False,
        }
    },
    location=LOCATION,
    dag=dag,
)


# Task Dependencies
create_dataset >> collect_data_task >> check_raw_data
check_raw_data >> create_features >> validate_features
validate_features >> generate_quality_report
generate_quality_report >> training_branch

# Conditional training path branch
training_branch >> [train_models_task, make_predictions_task]

# Make predictions
train_models_task >> make_predictions_task

# Evaluation and cleanup
make_predictions_task >> evaluate_predictions >> cleanup_old_predictions
