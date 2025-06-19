{{ config(materialized='table') }}

select
    pss.person_id,
    p.full_name,
    pss.season_year,
    pss.team_id,
    t.full_name as team_name,
    count(*) as games_played,
    avg(pss.num_minutes) as avg_minutes,
    avg(pss.points) as avg_points,
    avg(pss.assists) as avg_assists,
    avg(pss.rebounds_total) as avg_rebounds,
    avg(pss.steals) as avg_steals,
    avg(pss.blocks) as avg_blocks,
    
    sum(pss.field_goals_made)::float / nullif(sum(pss.field_goals_attempted)::float, 0) as season_fg_percentage,
    sum(pss.three_pointers_made)::float / nullif(sum(pss.three_pointers_attempted)::float, 0) as season_3p_percentage,
    sum(pss.free_throws_made)::float / nullif(sum(pss.free_throws_attempted)::float, 0) as season_ft_percentage,
    
    avg(pss.turnovers) as avg_turnovers,
    avg(pss.plus_minus_points) as avg_plus_minus,
    avg(pss.impact_rating_per_36_min) as avg_impact_rating,
    sum(pss.points) as total_points,
    sum(pss.assists) as total_assists,
    sum(pss.rebounds_total) as total_rebounds,
    sum(pss.steals) as total_steals,
    sum(pss.blocks) as total_blocks,
    current_timestamp as calculated_at
from {{ ref('stg_player_statistics') }} pss
join {{ ref('stg_players') }} p on pss.person_id = p.person_id
join {{ ref('stg_teams') }} t on pss.team_id = t.team_id
group by 1, 2, 3, 4, 5