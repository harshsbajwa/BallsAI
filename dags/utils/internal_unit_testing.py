"""
Internal unit testing utilities for DAG validation
"""
import logging
from airflow.models import DagBag

logger = logging.getLogger(__name__)

def assert_has_valid_dag(dag_module):
    """
    Assert that a DAG module contains valid DAGs.
    
    Args:
        dag_module: Python module containing DAG definitions
    """
    # Get all DAGs from the module
    dag_bag = DagBag(dag_folder=None, include_examples=False)
    
    # Import the module to trigger DAG creation
    if hasattr(dag_module, 'dag'):
        dag = dag_module.dag
        
        # Basic validation
        assert dag is not None, "DAG should not be None"
        assert dag.dag_id is not None, "DAG should have a dag_id"
        assert len(dag.tasks) > 0, "DAG should have at least one task"
        
        # Validate DAG structure
        dag_bag.bag_dag(dag, parent_dag=None, root_dag=None)
        
        # Check for cycles
        assert not dag.has_cycle(), "DAG should not have cycles"
        
        # Check for orphaned tasks
        assert not dag.get_orphaned_tasks(dag.tasks), "DAG should not have orphaned tasks"
        
        logger.info(f"DAG {dag.dag_id} passed validation with {len(dag.tasks)} tasks")
        
    else:
        raise ValueError("Module does not contain a 'dag' attribute")

def validate_task_dependencies(dag):
    """
    Validate that task dependencies make sense.
    """
    for task in dag.tasks:
        # Check upstream dependencies
        upstream_tasks = task.upstream_task_ids
        downstream_tasks = task.downstream_task_ids
        
        # Ensure no task depends on itself
        assert task.task_id not in upstream_tasks, f"Task {task.task_id} cannot depend on itself"
        assert task.task_id not in downstream_tasks, f"Task {task.task_id} cannot be downstream of itself"

def validate_dag_tags(dag, required_tags=None):
    """
    Validate that DAG has appropriate tags.
    """
    if required_tags:
        dag_tags = set(dag.tags or [])
        required_tags_set = set(required_tags)
        missing_tags = required_tags_set - dag_tags
        
        assert not missing_tags, f"DAG {dag.dag_id} missing required tags: {missing_tags}"
