from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocDeleteClusterOperator,
    DataprocSubmitPySparkJobOperator,
)
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

from data.dataproc import DataprocManager
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
def check_data_availability(**context):
    """Check if new data is available for processing."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=settings.PROJECT_ID)
    
    # Check for games that need feature engineering
    query = f"""
    SELECT COUNT(*) as pending_games
    FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.raw_games` g
    LEFT JOIN `{settings.PROJECT_ID}.{settings.DATASET_ID}.feature_set` f
    ON g.game_id = f.game_id
    WHERE g.game_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY)
    AND f.game_id IS NULL
    """
    
    result = client.query(query).to_dataframe()
    pending_games = result.iloc[0]['pending_games'] if not result.empty else 0
    
    logger.info(f"Found {pending_games} games pending feature engineering")
    
    return {
        'pending_games': pending_games,
        'should_process': pending_games > 0
    }

@safe_task_execution
def prepare_spark_job_config(**context):
    """Prepare configuration for Spark job."""
    check_result = context['task_instance'].xcom_pull(task_ids='check_data_availability')
    
    if not check_result.get('should_process', False):
        logger.info("No new data to process, skipping feature engineering")
        return None
    
    # Get incremental date
    incremental_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    spark_config = {
        'incremental_date': incremental_date,
        'cluster_name': f"nba-feature-eng-{context['ds_nodash']}",
        'job_args': [
            f"--project-id={settings.PROJECT_ID}",
            f"--incremental-date={incremental_date}",
            "--output-table=feature_set",
            "--mode=append"
        ]
    }
    
    logger.info(f"Spark job config: {spark_config}")
    return spark_config

# Create the DAG
dag = DAG(
    'nba_feature_engineering',
    default_args=default_args,
    description='NBA feature engineering with Dataproc',
    schedule_interval='0 12 * * *',  # Daily at noon
    catchup=False,
    max_active_runs=1,
    tags=['nba', 'feature-engineering', 'spark'],
)

# Define tasks
check_data_task = PythonOperator(
    task_id="check_data_availability",
    python_callable=check_data_availability,
    dag=dag,
)

prepare_config_task = PythonOperator(
    task_id="prepare_spark_job_config",
    python_callable=prepare_spark_job_config,
    dag=dag,
)

# Dataproc cluster operations
CLUSTER_NAME = "nba-feature-eng-{{ ds_nodash }}"

create_cluster_task = DataprocCreateClusterOperator(
    task_id="create_dataproc_cluster",
    project_id=settings.PROJECT_ID,
    cluster_config=settings.dataproc_cluster_config,
    region=settings.GCP_REGION,
    cluster_name=CLUSTER_NAME,
    dag=dag,
)

submit_spark_job_task = DataprocSubmitPySparkJobOperator(
    task_id="submit_spark",
    main=f"gs://{settings.GCS_BUCKET}/scripts/spark.py",
    cluster_name=CLUSTER_NAME,
    region=settings.GCP_REGION,
    project_id=settings.PROJECT_ID,
    dag=dag,
)

delete_cluster_task = DataprocDeleteClusterOperator(
    task_id="cleanup_dataproc_cluster",
    project_id=settings.PROJECT_ID,
    cluster_name=CLUSTER_NAME,
    region=settings.GCP_REGION,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# Set dependencies
check_data_task >> prepare_config_task >> create_cluster_task
create_cluster_task >> submit_spark_job_task >> delete_cluster_task