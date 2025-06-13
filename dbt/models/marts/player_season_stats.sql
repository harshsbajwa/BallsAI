{{ config(materialized='table') }}

select
    person_id,
    season_year,
    team_id,
    count(*) as games_played,
    avg(num_minutes) as avg_minutes,
    avg(points) as avg_points,
    avg(assists) as avg_assists,
    avg(rebounds_total) as avg_rebounds,
    avg(steals) as avg_steals,
    avg(blocks) as avg_blocks,
    avg(fg_percentage_calculated) as avg_fg_percentage,
    avg(three_pt_percentage_calculated) as avg_three_pt_percentage,
    avg(ft_percentage_calculated) as avg_ft_percentage,
    avg(turnovers) as avg_turnovers,
    avg(plus_minus_points) as avg_plus_minus,
    avg(player_efficiency_rating) as avg_per,
    sum(points) as total_points,
    sum(assists) as total_assists,
    sum(rebounds_total) as total_rebounds,
    sum(steals) as total_steals,
    sum(blocks) as total_blocks,
    current_timestamp as calculated_at
from {{ ref('stg_player_statistics') }}
group by person_id, season_year, team_id
