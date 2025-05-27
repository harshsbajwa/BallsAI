from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

from ml.training import NBAPredictor
from config.settings import settings

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'nba-betting-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['hsbajwah@gmail.com'],
}

def safe_task_execution(func):
    def wrapper(**context):
        try:
            start_time = datetime.now()
            result = func(**context)
            execution_time = datetime.now() - start_time
            logger.info(f"Task {func.__name__} completed successfully in {execution_time}")
            return result
        except Exception as e:
            logger.error(f"Task {func.__name__} failed: {e}")
            raise
    return wrapper

@safe_task_execution
def get_upcoming_games(**context):
    """Get upcoming games that need predictions."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=settings.PROJECT_ID)
    
    # Get games in next 3 days that have features but no predictions
    query = f"""
    SELECT 
        f.game_id,
        f.game_date,
        f.home_team_id,
        f.away_team_id,
        f.home_team_abbreviation,
        f.away_team_abbreviation
    FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.feature_set` f
    LEFT JOIN `{settings.PROJECT_ID}.{settings.DATASET_ID}.predictions` p
    ON f.game_id = p.game_id
    WHERE f.game_date BETWEEN CURRENT_DATE() AND DATE_ADD(CURRENT_DATE(), INTERVAL 3 DAY)
    AND p.game_id IS NULL
    AND f.feature_engineering_version = 'advanced_v1.0'
    ORDER BY f.game_date, f.game_id
    """
    
    result = client.query(query).to_dataframe()
    
    if result.empty:
        logger.info("No upcoming games need predictions")
        return []
    
    games = result.to_dict('records')
    logger.info(f"Found {len(games)} games needing predictions")
    
    return games

@safe_task_execution
def load_latest_model(**context):
    """Load the latest trained model."""
    try:
        # Try to load from GCS first, then local
        model_path = f"gs://{settings.GCS_BUCKET}/models/nba_models_latest.pth"
        predictor = NBAPredictor(model_path)
        
        model_info = predictor.get_model_info()
        logger.info(f"Loaded model: {model_info['model_version']} trained on {model_info['training_date']}")
        
        return {
            'model_loaded': True,
            'model_info': model_info
        }
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@safe_task_execution
def make_predictions(**context):
    """Make predictions for upcoming games."""
    games = context['task_instance'].xcom_pull(task_ids='get_upcoming_games')
    
    if not games:
        logger.info("No games to predict")
        return {"predictions_made": 0}
    
    # Load model
    predictor = NBAPredictor()
    
    # Get features for each game
    from google.cloud import bigquery
    client = bigquery.Client(project=settings.PROJECT_ID)
    
    predictions = []
    
    for game in games:
        try:
            # Get features for this game
            features_query = f"""
            SELECT * EXCEPT(game_id, game_date, home_team_id, away_team_id, 
                           home_team_abbreviation, away_team_abbreviation,
                           features_created_at, feature_engineering_version)
            FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.feature_set`
            WHERE game_id = '{game['game_id']}'
            """
            
            features_df = client.query(features_query).to_dataframe()
            
            if features_df.empty:
                logger.warning(f"No features found for game {game['game_id']}")
                continue
            
            # Convert to dict
            features_dict = features_df.iloc[0].to_dict()
            
            # Make prediction
            prediction = predictor.predict_game(features_dict)
            
            if prediction:
                prediction_record = {
                    'game_id': game['game_id'],
                    'game_date': game['game_date'],
                    'home_team_id': game['home_team_id'],
                    'away_team_id': game['away_team_id'],
                    'home_team_abbreviation': game['home_team_abbreviation'],
                    'away_team_abbreviation': game['away_team_abbreviation'],
                    'home_win_probability': prediction['home_win_probability'],
                    'away_win_probability': prediction['away_win_probability'],
                    'predicted_spread': prediction['predicted_spread'],
                    'spread_uncertainty': prediction['spread_uncertainty'],
                    'overall_confidence': prediction['overall_confidence'],
                    'prediction_strength': prediction['prediction_strength'],
                    'model_version': prediction['model_metadata']['model_version'],
                    'prediction_date': datetime.now().date(),
                    'prediction_timestamp': datetime.now()
                }
                
                predictions.append(prediction_record)
                
        except Exception as e:
            logger.error(f"Error predicting game {game['game_id']}: {e}")
            continue
    
    # Upload predictions to BigQuery
    if predictions:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = client.load_table_from_json(
            predictions,
            f"{settings.PROJECT_ID}.{settings.DATASET_ID}.predictions",
            job_config=job_config
        )
        job.result()
        
        logger.info(f"Successfully made {len(predictions)} predictions")
    
    return {"predictions_made": len(predictions)}

@safe_task_execution
def generate_prediction_report(**context):
    """Generate a summary report of predictions."""
    predictions_result = context['task_instance'].xcom_pull(task_ids='make_predictions')
    
    if predictions_result['predictions_made'] == 0:
        logger.info("No predictions to report")
        return {"report": "No predictions made"}
    
    from google.cloud import bigquery
    client = bigquery.Client(project=settings.PROJECT_ID)
    
    # Get today's predictions
    report_query = f"""
    SELECT 
        COUNT(*) as total_predictions,
        AVG(overall_confidence) as avg_confidence,
        COUNT(CASE WHEN prediction_strength = 'high' THEN 1 END) as high_confidence_predictions,
        COUNT(CASE WHEN home_win_probability > 0.6 THEN 1 END) as strong_home_favorites,
        COUNT(CASE WHEN ABS(predicted_spread) > 5 THEN 1 END) as large_spread_games
    FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.predictions`
    WHERE prediction_date = CURRENT_DATE()
    """
    
    result = client.query(report_query).to_dataframe()
    
    if not result.empty:
        report = result.iloc[0].to_dict()
        logger.info(f"Prediction report: {report}")
        return {"report": report}
    
    return {"report": "No report data available"}

# Create the DAG
dag = DAG(
    'nba_predictions',
    default_args=default_args,
    description='NBA daily predictions pipeline',
    schedule_interval='0 14 * * *',  # Daily at 2 PM
    catchup=False,
    max_active_runs=1,
    tags=['nba', 'predictions', 'daily'],
)

# Define tasks
get_games_task = PythonOperator(
    task_id="get_upcoming_games",
    python_callable=get_upcoming_games,
    dag=dag,
)

load_model_task = PythonOperator(
    task_id="load_latest_model",
    python_callable=load_latest_model,
    dag=dag,
)

make_predictions_task = PythonOperator(
    task_id="make_predictions",
    python_callable=make_predictions,
    dag=dag,
)

generate_report_task = PythonOperator(
    task_id="generate_prediction_report",
    python_callable=generate_prediction_report,
    dag=dag,
)

# Set dependencies
[get_games_task, load_model_task] >> make_predictions_task >> generate_report_task