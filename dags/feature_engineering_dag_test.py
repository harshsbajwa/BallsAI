import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import internal_unit_testing

def test_dag_import():
    from . import feature_engineering_dag
    internal_unit_testing.assert_has_valid_dag(feature_engineering_dag)

def test_dag_structure():
    from . import feature_engineering_dag
    
    dag = feature_engineering_dag.dag
    
    # Test specific DAG properties
    assert dag.dag_id == 'nba_feature_engineering'
    assert 'spark' in dag.tags
    
    # Test Dataproc tasks exist
    task_ids = [task.task_id for task in dag.tasks]
    assert 'create_dataproc_cluster' in task_ids
    assert 'cleanup_dataproc_cluster' in task_ids
