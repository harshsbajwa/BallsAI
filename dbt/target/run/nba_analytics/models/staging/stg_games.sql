
  create view "nba_pipeline"."staging"."stg_games__dbt_tmp"
    
    
  as (
    

select
    game_id,
    game_date,
    date_part('year', game_date) as season_year,
    date_part('month', game_date) as game_month,
    date_part('dow', game_date) as day_of_week,
    home_team_id,
    away_team_id,
    home_score,
    away_score,
    case 
        when home_score > away_score then home_team_id
        else away_team_id
    end as winning_team_id,
    case 
        when home_score > away_score then away_team_id
        else home_team_id
    end as losing_team_id,
    abs(home_score - away_score) as point_differential,
    game_type,
    attendance,
    arena_id,
    game_label,
    game_sub_label,
    series_game_number,
    created_at,
    updated_at
from "nba_pipeline"."public"."games"
  );