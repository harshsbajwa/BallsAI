import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from pathlib import Path


class Settings(BaseSettings):
    # Google Cloud settings - Core Infrastructure
    GOOGLE_APPLICATION_CREDENTIALS: str
    PROJECT_ID: str
    DATASET_ID: str = "nba_betting_data"
    GCS_BUCKET: Optional[str] = None
    GCP_REGION: str = "northamerica-northeast2" # Toronto
    
    # BigQuery Cost Optimization Settings
    BIGQUERY_LOCATION: str = "US"
    PARTITION_EXPIRATION_DAYS: int = 1825  # 5 years for games
    PBP_PARTITION_EXPIRATION_DAYS: int = 1095  # 3 years for play-by-play
    
    # Dataproc Cost Optimization Settings
    DATAPROC_CLUSTER_PREFIX: str = "nba-feature-eng"
    DATAPROC_MACHINE_TYPE: str = "n1-standard-4"
    DATAPROC_WORKER_COUNT: int = 2
    DATAPROC_PREEMPTIBLE: bool = True  # 80% cost savings
    DATAPROC_IMAGE_VERSION: str = "2.1-debian11"
    SPARK_BIGQUERY_CONNECTOR: str = "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2"
    
    # API Rate Limiting
    NBA_API_DELAY: float = 2.0  # Seconds between API calls
    NBA_API_MAX_RETRIES: int = 5
    NBA_API_BACKOFF_FACTOR: float = 2.0
    NBA_API_BATCH_SIZE: int = 25
    
    # Airflow settings
    AIRFLOW_HOME: str = "/opt/airflow"
    AIRFLOW_UID: int = 50000
    AIRFLOW_GID: int = 0
    USE_CLOUD_COMPOSER: bool = True  # vs self-managed

    # Database settings for self-managed Airflow
    POSTGRES_USER: str = "airflow"
    POSTGRES_PASSWORD: str = "airflow"
    POSTGRES_DB: str = "airflow"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432

    # Redis settings
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    
    # Model Training & Evaluation Settings
    VALIDATION_SPLIT: float = 0.2
    LEARNING_RATE: float = 0.001
    MAX_EPOCHS: int = 200
    EARLY_STOPPING_PATIENCE: int = 20
    BATCH_SIZE: int = 64
    
    # Model Performance Thresholds
    MIN_MODEL_ACCURACY: float = 0.52
    RETRAIN_FREQUENCY_DAYS: int = 7
    
    # Feature Engineering Windows
    ROLLING_WINDOWS: List[int] = [3, 5, 10]
    LOOKBACK_GAMES: int = 10

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore',
        case_sensitive=True
    )

    def __post_init__(self):
        if self.GCS_BUCKET is None:
            self.GCS_BUCKET = f"{self.PROJECT_ID}-nba-data"
        
        if not Path(self.GOOGLE_APPLICATION_CREDENTIALS).exists():
            raise FileNotFoundError(
                f"Service account key not found: {self.GOOGLE_APPLICATION_CREDENTIALS}"
            )

    @property
    def airflow_database_url(self) -> str:
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def celery_broker_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @property
    def dataproc_cluster_config(self) -> dict:
        """Cost-optimized Dataproc cluster configuration"""
        return {
            "master_config": {
                "num_instances": 1,
                "machine_type_uri": self.DATAPROC_MACHINE_TYPE,
                "disk_config": {
                    "boot_disk_type": "pd-standard",
                    "boot_disk_size_gb": 100
                },
                "is_preemptible": False  # Master should not be preemptible
            },
            "worker_config": {
                "num_instances": self.DATAPROC_WORKER_COUNT,
                "machine_type_uri": self.DATAPROC_MACHINE_TYPE,
                "disk_config": {
                    "boot_disk_type": "pd-standard", 
                    "boot_disk_size_gb": 100
                },
                "is_preemptible": self.DATAPROC_PREEMPTIBLE
            },
            "software_config": {
                "image_version": self.DATAPROC_IMAGE_VERSION,
                "properties": {
                    "spark:spark.jars.packages": self.SPARK_BIGQUERY_CONNECTOR,
                    "spark:spark.sql.adaptive.enabled": "true",
                    "spark:spark.sql.adaptive.coalescePartitions.enabled": "true"
                }
            },
            "gce_cluster_config": {
                "staging_bucket": self.GCS_BUCKET,
                "zone_uri": f"projects/{self.PROJECT_ID}/zones/{self.GCP_REGION}-a"
            }
        }

settings = Settings()
