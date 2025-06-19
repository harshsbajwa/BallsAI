{{
  config(
    materialized='table',
    tags=['spark_dependent']
  )
}}

select
    pss.person_id,
    pss.season_year,
    pss.full_name,
    pss.team_name,
    pss.avg_points,
    pss.avg_rebounds,
    pss.avg_assists,
    spark_scores.player_impact_rating as impact_score
from {{ ref('player_season_stats') }} pss
left join {{ source('analytics', 'analytics_player_impact_rating') }} spark_scores
    on pss.person_id = spark_scores.person_id
    and pss.season_year = spark_scores.season_year