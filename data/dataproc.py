from google.cloud import dataproc_v1
from google.cloud import storage
import logging
from config.settings import settings


logger = logging.getLogger(__name__)

class DataprocManager:
    def __init__(self):
        self.cluster_client = dataproc_v1.ClusterControllerClient(
            client_options={"api_endpoint": f"{settings.GCP_REGION}-dataproc.googleapis.com:443"}
        )
        self.job_client = dataproc_v1.JobControllerClient(
            client_options={"api_endpoint": f"{settings.GCP_REGION}-dataproc.googleapis.com:443"}
        )
        
    def create_ephemeral_cluster(self, cluster_name: str) -> str:
        """Create cost-optimized ephemeral cluster"""
        try:
            cluster_config = settings.dataproc_cluster_config
            
            cluster = {
                "project_id": settings.PROJECT_ID,
                "cluster_name": cluster_name,
                "config": cluster_config
            }
            
            operation = self.cluster_client.create_cluster(
                request={
                    "project_id": settings.PROJECT_ID,
                    "region": settings.GCP_REGION,
                    "cluster": cluster
                }
            )
            
            result = operation.result()
            logger.info(f"Created cluster {cluster_name} successfully")
            return cluster_name
            
        except Exception as e:
            logger.error(f"Error creating cluster {cluster_name}: {e}")
            raise
    
    def delete_cluster(self, cluster_name: str):
        """Delete cluster to save costs"""
        try:
            operation = self.cluster_client.delete_cluster(
                request={
                    "project_id": settings.PROJECT_ID,
                    "region": settings.GCP_REGION,
                    "cluster_name": cluster_name
                }
            )
            operation.result()
            logger.info(f"Deleted cluster {cluster_name}")
            
        except Exception as e:
            logger.error(f"Error deleting cluster {cluster_name}: {e}")
            # Don't raise - we want cleanup to proceed
    
    def submit_pyspark_job(self, cluster_name: str, main_python_file_uri: str, args: list = None):
        """Submit PySpark job to existing cluster"""
        try:
            job = {
                "placement": {"cluster_name": cluster_name},
                "pyspark_job": {
                    "main_python_file_uri": main_python_file_uri,
                    "args": args or [],
                    "properties": {
                        "spark.executor.memory": "3g",
                        "spark.executor.cores": "2",
                        "spark.dynamicAllocation.enabled": "true",
                        "spark.dynamicAllocation.minExecutors": "1",
                        "spark.dynamicAllocation.maxExecutors": "4"
                    }
                }
            }
            
            operation = self.job_client.submit_job(
                request={
                    "project_id": settings.PROJECT_ID,
                    "region": settings.GCP_REGION,
                    "job": job
                }
            )
            
            response = operation.result()
            logger.info(f"Job submitted successfully: {response.reference.job_id}")
            return response.reference.job_id
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            raise