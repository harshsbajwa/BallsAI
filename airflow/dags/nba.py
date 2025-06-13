from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import json
from minio import Minio
import logging

default_args = {
    'owner': 'nba-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'nba_daily_ingestion',
    default_args=default_args,
    description='Daily NBA data ingestion pipeline',
    schedule_interval='0 8 * * *',  # 8 AM
    catchup=False,
    max_active_runs=1,
)

def extract_nba_data(**context):
    """Extract NBA data using nba_api"""
    from src.utils.nba_client import NBADataClient
    
    client = NBADataClient()
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    todays_games = client.get_todays_games()
    
    recent_games = client.get_recent_games(days_back=3)
    
    boxscores = []
    for game in recent_games:
        game_id = game.get('GAME_ID')
        if game_id:
            boxscore = client.get_game_boxscore(game_id)
            if boxscore:
                boxscores.append({
                    'game_id': game_id,
                    'boxscore': boxscore,
                    'extraction_date': execution_date
                })
    
    minio_client = Minio(
        'minio-service:9000',
        access_key='minioadmin',
        secret_key='minioadmin',
        secure=False
    )
    
    bucket_name = 'nba-raw-data'
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    
    raw_data = {
        'todays_games': todays_games,
        'recent_games': recent_games,
        'boxscores': boxscores,
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    file_path = f"daily_extracts/{execution_date}/raw_data.json"
    minio_client.put_object(
        bucket_name,
        file_path,
        data=json.dumps(raw_data).encode('utf-8'),
        length=len(json.dumps(raw_data).encode('utf-8')),
        content_type='application/json'
    )
    
    logging.info(f"Successfully extracted and stored NBA data for {execution_date}")
    return file_path

def transform_and_load_data(**context):
    """Transform raw data and load into PostgreSQL"""
    from src.utils.data_transformer import DataTransformer
    
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    file_path = context['task_instance'].xcom_pull(task_ids='extract_nba_data')
    
    transformer = DataTransformer()
    
    minio_client = Minio(
        'minio-service:9000',
        access_key='minioadmin',
        secret_key='minioadmin',
        secure=False
    )
    
    raw_data_obj = minio_client.get_object('nba-raw-data', file_path)
    raw_data = json.loads(raw_data_obj.read().decode('utf-8'))
    
    transformer.process_daily_data(raw_data, execution_date)
    
    logging.info(f"Successfully transformed and loaded data for {execution_date}")

def run_dbt_transformations(**context):
    """Run dbt transformations"""
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    logging.info(f"Running dbt transformations for {execution_date}")
    
    return "dbt_ready"

def trigger_spark_analytics(**context):
    """Trigger PySpark analytics job"""
    from pyspark.sql import SparkSession
    from src.analytics.spark_jobs import run_daily_analytics
    
    execution_date = context['execution_date'].strftime('%Y-%m-%d')
    
    spark = SparkSession.builder \
        .appName(f"NBA_Analytics_{execution_date}") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    
    try:
        run_daily_analytics(spark, execution_date)
        logging.info(f"Successfully completed analytics for {execution_date}")
    finally:
        spark.stop()

extract_task = PythonOperator(
    task_id='extract_nba_data',
    python_callable=extract_nba_data,
    dag=dag,
)

transform_load_task = PythonOperator(
    task_id='transform_and_load_data',
    python_callable=transform_and_load_data,
    dag=dag,
)

dbt_run_task = BashOperator(
    task_id='run_dbt_transformations',
    bash_command='cd /opt/airflow/dbt && dbt run --target prod',
    dag=dag,
)

dbt_test_task = BashOperator(
    task_id='run_dbt_tests',
    bash_command='cd /opt/airflow/dbt && dbt test --target prod',
    dag=dag,
)

spark_analytics_task = PythonOperator(
    task_id='run_spark_analytics',
    python_callable=trigger_spark_analytics,
    dag=dag,
)

extract_task >> transform_load_task >> dbt_run_task >> dbt_test_task >> spark_analytics_task
