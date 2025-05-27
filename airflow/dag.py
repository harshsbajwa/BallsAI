from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocDeleteClusterOperator,
    DataprocSubmitPySparkJobOperator,
)
from airflow.providers.google.cloud.operators.bigquery import *
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

from data.collector import NBACollector
from data.schema import create_optimized_tables
from notebooks.models import AdvancedNBAModelTrainer
from setup.dataproc_setup import DataprocManager
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
    'email': ['hsbajwah@gmail.com'],
}

dag = DAG(
    'nba_betting_ml_pipeline',
    default_args=default_args,
    description='NBA betting ML pipeline',
    schedule_interval='0 10 * * *',  # Daily at 10 AM
    catchup=False,
    max_active_runs=1,
    tags=['nba', 'betting', 'ml', 'production'],
)

# Configuration
PROJECT_ID = settings.PROJECT_ID
DATASET_ID = settings.DATASET_ID
GCS_BUCKET = settings.GCS_BUCKET
GCP_REGION = settings.GCP_REGION
DATAPROC_CLUSTER_NAME = f"nba-feature-eng-{{{{ ds_nodash }}}}"

# Python Callables

def safe_task_execution(func):
    """Decorator for safe task execution"""
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
    """Initialize BigQuery tables"""
    logger.info("Initializing infrastructure...")
    
    client = create_optimized_tables()
    
    # Verify tables exist
    tables_to_check = [
        'raw_games', 'raw_player_boxscores_traditional', 
        'raw_player_boxscores_advanced', 'feature_set', 'predictions'
    ]
    
    for table_name in tables_to_check:
        try:
            table_ref = client.dataset(DATASET_ID).table(table_name)
            table = client.get_table(table_ref)
            logger.info(f"Verified table {table_name}: {table.num_rows} rows")
        except Exception as e:
            logger.error(f"Table {table_name} verification failed: {e}")
            raise
    
    return {"tables_verified": len(tables_to_check)}

@safe_task_execution
def get_last_processed_date(**context):
    """Get last processed date with fallback logic"""
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        
        # Try to get last processed date from metadata or actual data
        query = f"""
        SELECT MAX(game_date) as last_date
        FROM `{PROJECT_ID}.{DATASET_ID}.raw_games`
        WHERE load_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        """
        
        result = client.query(query).to_dataframe()
        
        if not result.empty and result.iloc[0]['last_date'] is not None:
            last_date = result.iloc[0]['last_date'].strftime('%Y-%m-%d')
            logger.info(f"Found last processed date: {last_date}")
        else:
            # Fallback to 30 days ago
            last_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            logger.info(f"Using fallback date: {last_date}")
        
        return last_date
        
    except Exception as e:
        logger.warning(f"Error getting last processed date: {e}")
        return (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

@safe_task_execution
def collect_nba_data(**context):
    """Data collection with comprehensive error handling"""
    from google.cloud import bigquery
    
    last_processed_date = context['task_instance'].xcom_pull(task_ids='get_last_processed_date_task')
    logger.info(f"Collecting NBA data since: {last_processed_date}")
    
    client = bigquery.Client(project=PROJECT_ID)
    collector = NBACollector(client)
    
    collection_summary = {
        'games_collected': 0,
        'player_stats_collected': 0,
        'errors_encountered': 0,
        'processing_time': None
    }
    
    try:
        start_time = datetime.now()
        
        # Collect recent games (last 7 days)
        recent_games = collector.collect_daily_games()
        if recent_games:
            collector.upload_to_bigquery("raw_games", recent_games)
            collection_summary['games_collected'] = len(recent_games)
            logger.info(f"Collected {len(recent_games)} recent games")
        
        # Collect completed game stats
        completed_game_ids = [
            game['game_id'] for game in recent_games 
            if game.get('game_status_id') == 3
        ]
        
        if completed_game_ids:
            # Process in batches for reliability
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
        
        # Generate collection report
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
    """Comprehensive data quality validation"""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=PROJECT_ID)
    
    validation_queries = {
        'games_today': f"""
            SELECT COUNT(*) as count
            FROM `{PROJECT_ID}.{DATASET_ID}.raw_games`
            WHERE game_date = CURRENT_DATE()
        """,
        'complete_games_today': f"""
            SELECT COUNT(*) as count  
            FROM `{PROJECT_ID}.{DATASET_ID}.raw_games`
            WHERE game_date = CURRENT_DATE()
            AND wl_home IS NOT NULL
        """,
        'player_stats_today': f"""
            SELECT COUNT(*) as count
            FROM `{PROJECT_ID}.{DATASET_ID}.raw_player_boxscores_traditional`
            WHERE DATE(load_timestamp) = CURRENT_DATE()
        """,
        'data_quality_score': f"""
            SELECT AVG(data_quality_score) as avg_quality
            FROM `{PROJECT_ID}.{DATASET_ID}.raw_games`
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
    
    # Check if we have sufficient data for feature engineering
    games_today = validation_results.get('games_today', 0)
    if games_today == 0:
        logger.warning("No games found for today - may skip feature engineering")
        return {'proceed_with_features': False, 'validation_results': validation_results}
    
    return {'proceed_with_features': True, 'validation_results': validation_results}

def check_feature_engineering_needed(**context):
    """Determine if feature engineering should proceed"""
    validation_result = context['task_instance'].xcom_pull(task_ids='validate_data_quality_task')
    
    if validation_result.get('proceed_with_features', False):
        return 'create_dataproc_cluster_task'
    else:
        return 'skip_feature_engineering_task'

@safe_task_execution
def create_ephemeral_dataproc_cluster(**context):
    """Create cost-optimized ephemeral Dataproc cluster"""
    dataproc_manager = DataprocManager()
    
    cluster_name = DATAPROC_CLUSTER_NAME.format(ds_nodash=context['ds_nodash'])
    
    try:
        created_cluster = dataproc_manager.create_ephemeral_cluster(cluster_name)
        logger.info(f"Created Dataproc cluster: {created_cluster}")
        return cluster_name
        
    except Exception as e:
        logger.error(f"Failed to create Dataproc cluster: {e}")
        raise

@safe_task_execution
def submit_spark_feature_engineering(**context):
    """Submit PySpark job for advanced feature engineering"""
    cluster_name = context['task_instance'].xcom_pull(task_ids='create_dataproc_cluster_task')
    last_processed_date = context['task_instance'].xcom_pull(task_ids='get_last_processed_date_task')
    
    dataproc_manager = DataprocManager()
    
    # PySpark script arguments
    spark_args = [
        f"--project-id={PROJECT_ID}",
        f"--incremental-date={last_processed_date}",
        "--output-table=feature_set",
        "--mode=append"
    ]
    
    script_uri = f"gs://{GCS_BUCKET}/scripts/spark_feature_engineering.py"
    
    try:
        job_id = dataproc_manager.submit_pyspark_job(cluster_name, script_uri, spark_args)
        logger.info(f"Submitted feature engineering job: {job_id}")
        return job_id
        
    except Exception as e:
        logger.error(f"Failed to submit PySpark job: {e}")
        raise

@safe_task_execution
def cleanup_dataproc_cluster(**context):
    """Delete ephemeral cluster to save costs"""
    cluster_name = context['task_instance'].xcom_pull(task_ids='create_dataproc_cluster_task')
    
    if cluster_name:
        dataproc_manager = DataprocManager()
        try:
            dataproc_manager.delete_cluster(cluster_name)
            logger.info(f"Deleted cluster: {cluster_name}")
        except Exception as e:
            logger.warning(f"Error deleting cluster {cluster_name}: {e}")
            # Don't raise - cleanup should continue

def check_model_retraining_needed(**context):
    """Intelligent model retraining decision"""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=PROJECT_ID)
    
    # Check recent model performance
    performance_query = f"""
    SELECT 
        AVG(CASE 
            WHEN p.home_win_probability > 0.5 AND g.wl_home = 'W' THEN 1
            WHEN p.home_win_probability <= 0.5 AND g.wl_home = 'L' THEN 1
            ELSE 0
        END) as recent_accuracy,
        COUNT(*) as prediction_count
    FROM `{PROJECT_ID}.{DATASET_ID}.predictions` p
    JOIN `{PROJECT_ID}.{DATASET_ID}.raw_games` g ON p.game_id = g.game_id
    WHERE p.prediction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
    AND g.wl_home IS NOT NULL
    """
    
    try:
        result = client.query(performance_query).to_dataframe()
        
        if not result.empty and result.iloc[0]['prediction_count'] > 10:
            recent_accuracy = result.iloc[0]['recent_accuracy']
            prediction_count = result.iloc[0]['prediction_count']
            
            logger.info(f"Recent model performance: {recent_accuracy:.3f} accuracy on {prediction_count} predictions")
            
            # Retrain if accuracy drops below threshold or weekly schedule
            should_retrain = (
                recent_accuracy < settings.MIN_MODEL_ACCURACY or
                datetime.now().weekday() == 0  # Monday retraining
            )
            
            if should_retrain:
                logger.info("Model retraining triggered")
                return 'train_models_task'
            else:
                logger.info("Model performance acceptable, skipping retraining")
                return 'make_predictions_task'
        else:
            logger.info("Insufficient prediction history, proceeding with training")
            return 'train_models_task'
            
    except Exception as e:
        logger.warning(f"Error checking model performance: {e}")
        return 'train_models_task'  # Default to training on error

@safe_task_execution
def train_models(**context):
    """Train models with evaluation"""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=PROJECT_ID)
    trainer = AdvancedNBAModelTrainer(client)
    
    try:
        model_data = trainer.train_all_models()
        
        # Store training metrics in BigQuery
        training_record = {
            'training_date': model_data['training_date'],
            'training_samples': model_data['training_samples'],
            'test_samples': model_data['test_samples'],
            'win_loss_accuracy': model_data['model_performance']['win_loss']['accuracy'],
            'win_loss_auc': model_data['model_performance']['win_loss']['auc'],
            'win_loss_log_loss': model_data['model_performance']['win_loss']['log_loss'],
            'win_loss_brier_score': model_data['model_performance']['win_loss']['brier_score'],
            'spread_mae': model_data['model_performance']['spread']['mae'],
            'spread_rmse': model_data['model_performance']['spread']['rmse'],
            'model_version': model_data['model_version'],
            'feature_count': len(model_data['feature_columns'])
        }
        
        # Upload to training history table
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = client.load_table_from_json(
            [training_record],
            f"{PROJECT_ID}.{DATASET_ID}.model_training_history",
            job_config=job_config
        )
        job.result()
        
        logger.info("Model training completed and logged")
        return {
            'model_path': model_data['model_path'],
            'performance_summary': model_data['model_performance']
        }
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

# Task Definitions with Configuration

# Infrastructure
initialize_infrastructure_task = PythonOperator(
    task_id="initialize_infrastructure_task",
    python_callable=initialize_infrastructure,
    dag=dag,
)

# Data Pipeline
get_last_processed_date_task = PythonOperator(
    task_id="get_last_processed_date_task",
    python_callable=get_last_processed_date,
    dag=dag,
)

collect_data_task = PythonOperator(
    task_id="collect_nba_data_task",
    python_callable=collect_nba_data,
    retries=3,
    retry_delay=timedelta(minutes=10),
    dag=dag,
)

validate_data_quality_task = PythonOperator(
    task_id="validate_data_quality_task",
    python_callable=validate_data_quality,
    dag=dag,
)

# Feature Engineering Branch
feature_engineering_branch = BranchPythonOperator(
    task_id="check_feature_engineering_needed_task",
    python_callable=check_feature_engineering_needed,
    dag=dag,
)

skip_feature_engineering_task = PythonOperator(
    task_id="skip_feature_engineering_task",
    python_callable=lambda: logger.info("Skipping feature engineering - no new data"),
    dag=dag,
)

# Dataproc Operations
create_dataproc_cluster_task = DataprocCreateClusterOperator(
    task_id="create_dataproc_cluster_task",
    project_id=PROJECT_ID,
    cluster_config=settings.dataproc_cluster_config,
    region=GCP_REGION,
    cluster_name=DATAPROC_CLUSTER_NAME,
    dag=dag,
)

submit_spark_job_task = DataprocSubmitPySparkJobOperator(
    task_id="submit_spark_feature_engineering_task",
    main=f"gs://{GCS_BUCKET}/scripts/spark_feature_engineering.py",
    cluster_name=DATAPROC_CLUSTER_NAME,
    region=GCP_REGION,
    project_id=PROJECT_ID,
    dag=dag,
)

delete_dataproc_cluster_task = DataprocDeleteClusterOperator(
    task_id="cleanup_dataproc_cluster_task",
    project_id=PROJECT_ID,
    cluster_name=DATAPROC_CLUSTER_NAME,
    region=GCP_REGION,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# Model Training Branch
model_training_branch = BranchPythonOperator(
    task_id="check_model_retraining_needed_task",
    python_callable=check_model_retraining_needed,
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id="train_models_task",
    python_callable=train_models,
    execution_timeout=timedelta(hours=3),
    dag=dag,
)

# Prediction and Evaluation
make_predictions_task = PythonOperator(
    task_id="make_predictions_task",
    python_callable=lambda: logger.info("Making predictions (implementation needed)"),
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

# Task Dependencies
initialize_infrastructure_task >> get_last_processed_date_task
get_last_processed_date_task >> collect_data_task >> validate_data_quality_task
validate_data_quality_task >> feature_engineering_branch

# Feature Engineering Path
feature_engineering_branch >> [create_dataproc_cluster_task, skip_feature_engineering_task]
create_dataproc_cluster_task >> submit_spark_job_task >> delete_dataproc_cluster_task

# Model Training Path
[delete_dataproc_cluster_task, skip_feature_engineering_task] >> model_training_branch
model_training_branch >> [train_models_task, make_predictions_task]