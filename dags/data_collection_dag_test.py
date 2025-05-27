import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import internal_unit_testing

def test_dag_import():
    from . import data_collection_dag
    internal_unit_testing.assert_has_valid_dag(data_collection_dag)

def test_dag_structure():
    from . import data_collection_dag
    
    dag = data_collection_dag.dag
    
    # Test specific DAG properties
    assert dag.dag_id == 'nba_data_collection'
    assert dag.schedule_interval == '0 10 * * *'
    assert 'nba' in dag.tags
    assert 'data-collection' in dag.tags
    
    # Test task dependencies
    internal_unit_testing.validate_task_dependencies(dag)
    