

select
    person_id,
    first_name,
    last_name,
    concat(first_name, ' ', last_name) as full_name,
    birthdate,
    last_attended,
    country,
    height,
    body_weight,
    case 
        when guard then 'Guard'
        when forward then 'Forward'
        when center then 'Center'
        else 'Unknown'
    end as primary_position,
    draft_year,
    draft_round,
    draft_number,
    created_at,
    updated_at
from "nba_pipeline"."public"."players"