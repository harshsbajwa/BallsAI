import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    # Google Cloud settings
    GOOGLE_APPLICATION_CREDENTIALS: str
    PROJECT_ID: str
    DATASET_ID: str = "nba_betting_data"
    GCS_BUCKET: Optional[str] = None

    # Airflow settings
    AIRFLOW_HOME: str = "/opt/airflow"
    AIRFLOW_UID: int = 50000
    AIRFLOW_GID: int = 0

    # Database settings
    POSTGRES_USER: str = "airflow"
    POSTGRES_PASSWORD: str = "airflow"
    POSTGRES_DB: str = "airflow"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432

    # Redis settings
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    # Data collection settings
    max_retries: int = 3
    retry_delay: int = 30
    batch_size: int = 64
    
    # Model settings
    validation_split: float = 0.2
    learning_rate: float = 0.001
    max_epochs: int = 200
    early_stopping_patience: int = 20

    # Prediction settings
    confidence_threshold: float = 0.7
    max_bet_percentage: float = 0.05

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore',
        case_sensitive=True
    )

    def __post_init__(self):
        """Validate settings after initialization"""
        if self.GCS_BUCKET is None:
            self.GCS_BUCKET = f"{self.PROJECT_ID}-nba-data"
        
        # Validate paths
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
    def celery_result_backend(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

settings = Settings()
