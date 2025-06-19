from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import json
import io
from minio import Minio
import logging
import os

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
    schedule='0 8 * * *',  # 8 AM
    catchup=False,
    max_active_runs=1,
)

def extract_nba_data(**context):
    """Extract NBA data using nba_api"""
    from src.utils.client import NBADataClient
    
    client = NBADataClient()
    execution_date = context['logical_date'].strftime('%Y-%m-%d')
    
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
    
    minio_host = os.getenv("MINIO_HOST", "minio-service")
    minio_client = Minio(
        f'{minio_host}:9000',
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
    data_bytes = json.dumps(raw_data).encode('utf-8')
    minio_client.put_object(
        bucket_name,
        file_path,
        data=io.BytesIO(data_bytes),
        length=len(data_bytes),
        content_type='application/json'
    )
    
    logging.info(f"Successfully extracted and stored NBA data for {execution_date}")
    return file_path

def transform_and_load_data(**context):
    """Transform raw data and load into PostgreSQL"""
    from src.utils.transformer import DataTransformer
    
    execution_date = context['logical_date'].strftime('%Y-%m-%d')
    file_path = context['task_instance'].xcom_pull(task_ids='extract_nba_data')
    
    transformer = DataTransformer()
    
    minio_host = os.getenv("MINIO_HOST", "minio-service")
    minio_client = Minio(
        f'{minio_host}:9000',
        access_key='minioadmin',
        secret_key='minioadmin',
        secure=False
    )
    
    raw_data_obj = minio_client.get_object('nba-raw-data', file_path)
    raw_data = json.loads(raw_data_obj.read().decode('utf-8'))
    
    transformer.process_daily_data(raw_data, execution_date)
    
    logging.info(f"Successfully transformed and loaded data for {execution_date}")

def trigger_spark_job(**context):
    """Trigger PySpark Player Impact Rating Calculation job"""
    from pyspark.sql import SparkSession
    from src.analytics.spark_jobs import run_player_impact_rating_job
    
    spark = SparkSession.builder \
        .appName(f"PIR_Calculation_{context['ds_nodash']}") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3") \
        .config("spark.driver.memory", "2g") \
        .master("local[*]") \
        .getOrCreate()
    
    try:
        run_player_impact_rating_job(spark)
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

dbt_base_run = BashOperator(
    task_id='dbt_base_run',
    bash_command="cd /opt/airflow/dbt && dbt run --profiles-dir . --target prod --exclude tag:spark_dependent"
)

spark_analytics_task = PythonOperator(
    task_id='run_spark_player_impact_job',
    python_callable=trigger_spark_job,
)

dbt_final_run = BashOperator(
    task_id='dbt_final_run',
    bash_command="cd /opt/airflow/dbt && dbt run --profiles-dir . --target prod --select tag:spark_dependent"
)

dbt_test = BashOperator(
    task_id='dbt_test',
    bash_command="cd /opt/airflow/dbt && dbt test --profiles-dir . --target prod"
)

extract_task >> transform_load_task >> dbt_base_run >> spark_analytics_task >> dbt_final_run >> dbt_test