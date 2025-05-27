from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

from data.schema import create_optimized_tables
from config.settings import settings

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'nba-betting-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
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
def initialize_infrastructure(**context):
    """Initialize BigQuery tables and infrastructure."""
    logger.info("Initializing infrastructure...")
    
    client = create_optimized_tables()
    
    tables_to_check = [
        'raw_games', 'raw_player_boxscores_traditional',
        'raw_player_boxscores_advanced', 'feature_set', 'predictions'
    ]
    
    for table_name in tables_to_check:
        try:
            table_ref = client.dataset(settings.DATASET_ID).table(table_name)
            table = client.get_table(table_ref)
            logger.info(f"Verified table {table_name}: {table.num_rows} rows")
        except Exception as e:
            logger.error(f"Table {table_name} verification failed: {e}")
            raise
    
    return {"tables_verified": len(tables_to_check)}

@safe_task_execution
def validate_infrastructure(**context):
    """Validate infrastructure health."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=settings.PROJECT_ID)
    
    validation_queries = {
        'dataset_exists': f"""
            SELECT COUNT(*) as table_count
            FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.__TABLES__`
        """,
        'recent_data': f"""
            SELECT COUNT(*) as recent_games
            FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.raw_games`
            WHERE game_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        """
    }
    
    results = {}
    for check_name, query in validation_queries.items():
        try:
            result = client.query(query).to_dataframe()
            results[check_name] = result.iloc[0].values[0] if not result.empty else 0
        except Exception as e:
            logger.warning(f"Validation check {check_name} failed: {e}")
            results[check_name] = 'error'
    
    logger.info(f"Infrastructure validation: {results}")
    return results

# Create the DAG
dag = DAG(
    'nba_infrastructure',
    default_args=default_args,
    description='NBA infrastructure setup and validation',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    max_active_runs=1,
    tags=['nba', 'infrastructure', 'setup'],
)

# Define tasks
initialize_task = PythonOperator(
    task_id="initialize_infrastructure",
    python_callable=initialize_infrastructure,
    dag=dag,
)

validate_task = PythonOperator(
    task_id="validate_infrastructure",
    python_callable=validate_infrastructure,
    dag=dag,
)

# Set dependencies
initialize_task >> validate_task