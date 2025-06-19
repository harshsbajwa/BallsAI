
  create view "nba_pipeline"."staging"."stg_teams__dbt_tmp"
    
    
  as (
    

select
    team_id,
    team_city,
    team_name,
    concat(team_city, ' ', team_name) as full_name,
    team_abbrev,
    season_founded,
    season_active_till,
    league,
    created_at,
    updated_at
from "nba_pipeline"."public"."teams"
  );