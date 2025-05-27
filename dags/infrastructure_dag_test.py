import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import internal_unit_testing

def test_dag_import():
    from . import infrastructure_dag
    internal_unit_testing.assert_has_valid_dag(infrastructure_dag)

def test_dag_structure():
    from . import infrastructure_dag
    
    dag = infrastructure_dag.dag
    
    # Test specific DAG properties
    assert dag.dag_id == 'nba_infrastructure'
    assert 'nba' in dag.tags
    assert 'infrastructure' in dag.tags
    
    # Test task count
    assert len(dag.tasks) == 2
    
    # Test task IDs
    task_ids = [task.task_id for task in dag.tasks]
    expected_tasks = ['initialize_infrastructure', 'validate_infrastructure']
    assert all(task_id in task_ids for task_id in expected_tasks)
    