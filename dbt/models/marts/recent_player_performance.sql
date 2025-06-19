{{ config(materialized='table') }}

select
    p.person_id,
    p.full_name,
    ps.team_id,
    ps.game_date,
    ps.points,
    ps.assists,
    ps.rebounds_total,
    ps.steals,
    ps.blocks,
    ps.fg_percentage_calculated,
    ps.three_pt_percentage_calculated,
    -- Rolling averages (last 10 games)
    avg(ps.points) over (
        partition by ps.person_id 
        order by ps.game_date 
        rows between 9 preceding and current row
    ) as points_10_game_avg,
    avg(ps.assists) over (
        partition by ps.person_id 
        order by ps.game_date 
        rows between 9 preceding and current row
    ) as assists_10_game_avg,
    avg(ps.rebounds_total) over (
        partition by ps.person_id 
        order by ps.game_date 
        rows between 9 preceding and current row
    ) as rebounds_10_game_avg,
    row_number() over (
        partition by ps.person_id 
        order by ps.game_date desc
    ) as recency_rank
from {{ ref('stg_player_statistics') }} ps
join {{ ref('stg_players') }} p on ps.person_id = p.person_id
where ps.game_date >= current_date - interval '30 days'