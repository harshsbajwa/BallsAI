
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select season_year
from "nba_pipeline"."analytics"."analytics_player_impact_rating"
where season_year is null



  
  
      
    ) dbt_internal_test