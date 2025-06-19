

select
    p.person_id,
    p.full_name,
    p.primary_position,
    p.draft_year,
    p.draft_number,
    pir.season_year,
    pir.player_impact_rating,
    pir.points_per_36,
    pir.rebounds_per_36,
    pir.assists_per_36,
    pir.avg_ts_percentage,
    current_timestamp as calculated_at
from "nba_pipeline"."staging"."stg_players" p
join "nba_pipeline"."analytics"."analytics_player_impact_rating" pir
    on p.person_id = pir.person_id