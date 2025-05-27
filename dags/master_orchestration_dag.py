from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'owner': 'nba-betting-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'nba_master_orchestration',
    default_args=default_args,
    description='Master orchestration for NBA betting pipeline',
    schedule_interval='0 9 * * *',  # Daily at 9 AM
    catchup=False,
    tags=['nba', 'orchestration', 'master'],
)

# Trigger data collection
trigger_data_collection = TriggerDagRunOperator(
    task_id='trigger_data_collection',
    trigger_dag_id='nba_data_collection',
    dag=dag,
)

# Trigger feature engineering (with delay)
trigger_feature_engineering = TriggerDagRunOperator(
    task_id='trigger_feature_engineering',
    trigger_dag_id='nba_feature_engineering',
    execution_date_fn=lambda dt: dt + timedelta(hours=2),
    dag=dag,
)

# Trigger predictions (with delay)
trigger_predictions = TriggerDagRunOperator(
    task_id='trigger_predictions',
    trigger_dag_id='nba_predictions',
    execution_date_fn=lambda dt: dt + timedelta(hours=4),
    dag=dag,
)

# Set dependencies
trigger_data_collection >> trigger_feature_engineering >> trigger_predictions