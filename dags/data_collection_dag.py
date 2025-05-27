from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

from data.collector import NBACollector
from config.settings import settings

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'nba-betting-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
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
def get_last_processed_date(**context):
    """Get the last processed date for incremental collection."""
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=settings.PROJECT_ID)
        
        query = f"""
        SELECT MAX(game_date) as last_date
        FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.raw_games`
        WHERE load_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        """
        
        result = client.query(query).to_dataframe()
        
        if not result.empty and result.iloc[0]['last_date'] is not None:
            last_date = result.iloc[0]['last_date'].strftime('%Y-%m-%d')
            logger.info(f"Found last processed date: {last_date}")
        else:
            last_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            logger.info(f"Using fallback date: {last_date}")
        
        return last_date
    except Exception as e:
        logger.warning(f"Error getting last processed date: {e}")
        return (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

@safe_task_execution
def collect_nba_data(**context):
    """Collect NBA games and player statistics."""
    from google.cloud import bigquery
    
    last_processed_date = context['task_instance'].xcom_pull(task_ids='get_last_processed_date')
    logger.info(f"Collecting NBA data since: {last_processed_date}")
    
    client = bigquery.Client(project=settings.PROJECT_ID)
    collector = NBACollector(client)
    
    collection_summary = {
        'games_collected': 0,
        'player_stats_collected': 0,
        'errors_encountered': 0,
        'processing_time': None
    }
    
    try:
        start_time = datetime.now()
        
        # Collect recent games
        recent_games = collector.collect_daily_games()
        if recent_games:
            collector.upload_to_bigquery("raw_games", recent_games)
            collection_summary['games_collected'] = len(recent_games)
            logger.info(f"Collected {len(recent_games)} recent games")
        
        # Collect player stats for completed games
        completed_game_ids = [
            game['game_id'] for game in recent_games
            if game.get('game_status_id') == 3
        ]
        
        if completed_game_ids:
            batch_size = settings.NBA_API_BATCH_SIZE
            total_player_stats = []
            
            for i in range(0, len(completed_game_ids), batch_size):
                batch = completed_game_ids[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} games")
                
                player_stats = collector.collect_box_scores_batch(batch)
                if player_stats:
                    total_player_stats.extend(player_stats)
            
            if total_player_stats:
                collector.upload_to_bigquery("raw_player_boxscores_traditional", total_player_stats)
                collection_summary['player_stats_collected'] = len(total_player_stats)
        
        report = collector.generate_collection_report()
        collection_summary['errors_encountered'] = len(report.get('failed_operations', []))
        collection_summary['processing_time'] = str(datetime.now() - start_time)
        
        logger.info(f"Data collection summary: {collection_summary}")
        return collection_summary
    
    except Exception as e:
        logger.error(f"Critical error in data collection: {e}")
        collection_summary['errors_encountered'] += 1
        raise

@safe_task_execution
def validate_data_quality(**context):
    """Validate the quality of collected data."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=settings.PROJECT_ID)
    
    validation_queries = {
        'games_today': f"""
            SELECT COUNT(*) as count
            FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.raw_games`
            WHERE game_date = CURRENT_DATE()
        """,
        'complete_games_today': f"""
            SELECT COUNT(*) as count
            FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.raw_games`
            WHERE game_date = CURRENT_DATE()
            AND wl_home IS NOT NULL
        """,
        'player_stats_today': f"""
            SELECT COUNT(*) as count
            FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.raw_player_boxscores_traditional`
            WHERE DATE(load_timestamp) = CURRENT_DATE()
        """,
        'data_quality_score': f"""
            SELECT AVG(data_quality_score) as avg_quality
            FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.raw_games`
            WHERE game_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        """
    }
    
    validation_results = {}
    
    for check_name, query in validation_queries.items():
        try:
            result = client.query(query).to_dataframe()
            validation_results[check_name] = result.iloc[0].values[0] if not result.empty else 0
        except Exception as e:
            logger.warning(f"Validation check {check_name} failed: {e}")
            validation_results[check_name] = 'error'
    
    logger.info(f"Data quality validation: {validation_results}")
    
    games_today = validation_results.get('games_today', 0)
    proceed_with_features = games_today > 0
    
    return {
        'proceed_with_features': proceed_with_features,
        'validation_results': validation_results
    }

# Create the DAG
dag = DAG(
    'nba_data_collection',
    default_args=default_args,
    description='NBA data collection pipeline',
    schedule_interval='0 10 * * *',  # Daily at 10 AM
    catchup=False,
    max_active_runs=1,
    tags=['nba', 'data-collection', 'daily'],
)

# Define tasks
get_last_date_task = PythonOperator(
    task_id="get_last_processed_date",
    python_callable=get_last_processed_date,
    dag=dag,
)

collect_data_task = PythonOperator(
    task_id="collect_nba_data",
    python_callable=collect_nba_data,
    dag=dag,
)

validate_quality_task = PythonOperator(
    task_id="validate_data_quality",
    python_callable=validate_data_quality,
    dag=dag,
)

# Set dependencies
get_last_date_task >> collect_data_task >> validate_quality_task