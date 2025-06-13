{{ config(materialized='table') }}

select
    team_id,
    season_year,
    count(*) as games_played,
    sum(case when win then 1 else 0 end) as wins,
    sum(case when not win then 1 else 0 end) as losses,
    sum(case when win then 1 else 0 end)::float / count(*)::float as win_percentage,
    avg(team_score) as avg_points_scored,
    avg(opponent_score) as avg_points_allowed,
    avg(team_score - opponent_score) as avg_point_differential,
    avg(assists) as avg_assists,
    avg(rebounds_total) as avg_rebounds,
    avg(steals) as avg_steals,
    avg(blocks) as avg_blocks,
    avg(turnovers) as avg_turnovers,
    current_timestamp as calculated_at
from {{ source('public', 'team_statistics') }}
group by team_id, season_year
