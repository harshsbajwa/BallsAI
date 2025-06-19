
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select game_id
from "nba_pipeline"."staging"."stg_games"
where game_id is null



  
  
      
    ) dbt_internal_test