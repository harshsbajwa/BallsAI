
  create view "nba_pipeline"."staging"."stg_team_statistics__dbt_tmp"
    
    
  as (
    

select
    id,
    game_id,
    team_id,
    opponent_team_id,
    game_date,
    date_part('year', game_date) as season_year,
    home,
    win,
    team_score,
    opponent_score,
    assists,
    blocks,
    steals,
    field_goals_attempted,
    field_goals_made,
    field_goals_percentage,
    three_pointers_attempted,
    three_pointers_made,
    three_pointers_percentage,
    free_throws_attempted,
    free_throws_made,
    free_throws_percentage,
    rebounds_defensive,
    rebounds_offensive,
    rebounds_total,
    fouls_personal,
    turnovers,
    plus_minus_points,
    created_at
from "nba_pipeline"."public"."team_statistics"
  );