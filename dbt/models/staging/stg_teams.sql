{{ config(materialized='view') }}

select
    team_id,
    team_city,
    team_name,
    concat(team_city, ' ', team_name) as full_name,
    team_abbrev,
    season_founded,
    season_active_till,
    league,
    created_at,
    updated_at
from {{ source('public', 'teams') }}