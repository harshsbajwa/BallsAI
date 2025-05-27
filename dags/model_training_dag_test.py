import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import internal_unit_testing

def test_dag_import():
    from . import model_training_dag
    internal_unit_testing.assert_has_valid_dag(model_training_dag)

def test_dag_structure():
    from . import model_training_dag
    
    dag = model_training_dag.dag
    
    # Test specific DAG properties
    assert dag.dag_id == 'nba_model_training'
    assert dag.schedule_interval == '0 2 * * 1'  # Weekly
    assert 'ml' in dag.tags
    
    # Test branching logic exists
    task_ids = [task.task_id for task in dag.tasks]
    assert 'check_model_retraining_needed' in task_ids
