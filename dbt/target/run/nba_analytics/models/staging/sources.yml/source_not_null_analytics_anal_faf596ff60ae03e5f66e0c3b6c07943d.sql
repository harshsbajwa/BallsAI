
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select person_id
from "nba_pipeline"."analytics"."analytics_player_impact_rating"
where person_id is null



  
  
      
    ) dbt_internal_test