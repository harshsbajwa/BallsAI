from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

from ml.training import AdvancedNBAModelTrainer
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

def check_model_retraining_needed(**context):
    """Check if model retraining is needed based on performance."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=settings.PROJECT_ID)
    
    # Check recent model performance
    performance_query = f"""
    SELECT
        AVG(CASE
            WHEN p.home_win_probability > 0.5 AND g.wl_home = 'W' THEN 1
            WHEN p.home_win_probability <= 0.5 AND g.wl_home = 'L' THEN 1
            ELSE 0
        END) as recent_accuracy,
        COUNT(*) as prediction_count
    FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.predictions` p
    JOIN `{settings.PROJECT_ID}.{settings.DATASET_ID}.raw_games` g ON p.game_id = g.game_id
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
                return 'skip_training_task'
        else:
            logger.info("Insufficient prediction history, proceeding with training")
            return 'train_models_task'
    
    except Exception as e:
        logger.warning(f"Error checking model performance: {e}")
        return 'train_models_task'  # Default to training on error

@safe_task_execution
def train_models(**context):
    """Train the NBA prediction models."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=settings.PROJECT_ID)
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
            f"{settings.PROJECT_ID}.{settings.DATASET_ID}.model_training_history",
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

@safe_task_execution
def skip_training(**context):
    """Skip training task."""
    logger.info("Skipping model training - performance acceptable")
    return {"status": "skipped", "reason": "performance_acceptable"}

@safe_task_execution
def validate_models(**context):
    """Validate trained models."""
    training_result = context['task_instance'].xcom_pull(task_ids='train_models_task')
    
    if training_result:
        performance = training_result.get('performance_summary', {})
        win_loss_performance = performance.get('win_loss', {})
        
        # Validate minimum performance thresholds
        accuracy = win_loss_performance.get('accuracy', 0)
        auc = win_loss_performance.get('auc', 0)
        
        if accuracy >= settings.MIN_MODEL_ACCURACY and auc >= 0.55:
            logger.info(f"Model validation passed - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
            return {"validation_passed": True, "accuracy": accuracy, "auc": auc}
        else:
            logger.warning(f"Model validation failed - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
            raise ValueError("Model performance below acceptable thresholds")
    else:
        logger.info("No training performed, skipping validation")
        return {"validation_passed": True, "reason": "no_training"}

# Create the DAG
dag = DAG(
    'nba_model_training',
    default_args=default_args,
    description='NBA model training pipeline',
    schedule_interval='0 2 * * 1',  # Weekly on Monday at 2 AM
    catchup=False,
    max_active_runs=1,
    tags=['nba', 'model-training', 'ml'],
)

# Define tasks
check_retraining_task = BranchPythonOperator(
    task_id="check_model_retraining_needed",
    python_callable=check_model_retraining_needed,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id="train_models_task",
    python_callable=train_models,
    execution_timeout=timedelta(hours=3),
    dag=dag,
)

skip_training_task = PythonOperator(
    task_id="skip_training_task",
    python_callable=skip_training,
    dag=dag,
)

validate_models_task = PythonOperator(
    task_id="validate_models",
    python_callable=validate_models,
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

# Set dependencies
check_retraining_task >> [train_models_task, skip_training_task]
[train_models_task, skip_training_task] >> validate_models_task