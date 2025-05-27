"""
Common utilities for NBA DAGs
"""
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def safe_task_execution(func):
    """
    Decorator to safely execute tasks with error handling and logging.
    """
    @wraps(func)
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

def get_bigquery_client():
    """Get BigQuery client with proper configuration."""
    from google.cloud import bigquery
    from config.settings import settings
    
    return bigquery.Client(project=settings.PROJECT_ID)

def log_task_metrics(task_name: str, metrics: Dict[str, Any]):
    """
    Log task execution metrics in a standardized format.
    """
    metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
    logger.info(f"[{task_name}] Metrics - {metrics_str}")

def validate_required_env_vars():
    """
    Validate that required environment variables are set.
    """
    from config.settings import settings
    
    required_vars = [
        'PROJECT_ID',
        'DATASET_ID',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not hasattr(settings, var) or not getattr(settings, var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

def get_xcom_value(context: Dict, task_id: str, key: Optional[str] = None):
    """
    Safely get XCom value with error handling.
    """
    try:
        return context['task_instance'].xcom_pull(task_ids=task_id, key=key)
    except Exception as e:
        logger.warning(f"Failed to get XCom value from {task_id}: {e}")
        return None