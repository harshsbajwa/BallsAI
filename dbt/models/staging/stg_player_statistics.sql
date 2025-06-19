{{ config(materialized='view') }}

select
    id,
    person_id,
    game_id,
    game_date,
    date_part('year', game_date) as season_year,
    team_id,
    opponent_team_id,
    win,
    home,
    num_minutes,
    points,
    assists,
    blocks,
    steals,
    field_goals_attempted,
    field_goals_made,
    case 
        when field_goals_attempted > 0 
        then field_goals_made::float / field_goals_attempted::float
        else 0
    end as fg_percentage_calculated,
    three_pointers_attempted,
    three_pointers_made,
    case 
        when three_pointers_attempted > 0 
        then three_pointers_made::float / three_pointers_attempted::float
        else 0
    end as three_pt_percentage_calculated,
    free_throws_attempted,
    free_throws_made,
    case 
        when free_throws_attempted > 0 
        then free_throws_made::float / free_throws_attempted::float
        else 0
    end as ft_percentage_calculated,
    rebounds_defensive,
    rebounds_offensive,
    rebounds_total,
    fouls_personal,
    turnovers,
    plus_minus_points,
    -- Advanced stats
    case 
        when num_minutes > 0 
        then (points + rebounds_total + assists + steals + blocks - 
              (field_goals_attempted - field_goals_made) - 
              (free_throws_attempted - free_throws_made) - turnovers) / num_minutes * 36
        else 0
    end as impact_rating_per_36_min,
    created_at
from {{ source('public', 'player_statistics') }}