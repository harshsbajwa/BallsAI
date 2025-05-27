import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import internal_unit_testing

def test_dag_import():
    from . import prediction_dag
    internal_unit_testing.assert_has_valid_dag(prediction_dag)

def test_dag_structure():
    from . import prediction_dag
    
    dag = prediction_dag.dag
    
    # Test specific DAG properties
    assert dag.dag_id == 'nba_predictions'
    assert dag.schedule_interval == '0 14 * * *'
    assert 'predictions' in dag.tags
    
    # Test required tasks exist
    task_ids = [task.task_id for task in dag.tasks]
    expected_tasks = ['get_upcoming_games', 'load_latest_model', 'make_predictions']
    assert all(task_id in task_ids for task_id in expected_tasks)
