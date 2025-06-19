

with game_pairs as (
    select
        game_id,
        game_date,
        least(home_team_id, away_team_id) as team_1_id,
        greatest(home_team_id, away_team_id) as team_2_id,
        
        case
            when winning_team_id = least(home_team_id, away_team_id) then 1
            else 0
        end as team_1_win,
        case
            when winning_team_id = greatest(home_team_id, away_team_id) then 1
            else 0
        end as team_2_win,
        
        point_differential
        
    from "nba_pipeline"."staging"."stg_games"
    where winning_team_id is not null
)

select
    team_1_id,
    team_2_id,
    sum(team_1_win) as team_1_wins,
    sum(team_2_win) as team_2_wins,
    count(*) as total_games,
    avg(point_differential) as avg_margin,
    max(game_date) as last_meeting,
    current_timestamp as calculated_at
from game_pairs
group by 1, 2