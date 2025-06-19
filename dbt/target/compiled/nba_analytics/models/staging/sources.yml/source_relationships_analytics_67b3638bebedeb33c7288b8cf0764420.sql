
    
    

with child as (
    select person_id as from_field
    from "nba_pipeline"."analytics"."analytics_player_impact_rating"
    where person_id is not null
),

parent as (
    select person_id as to_field
    from "nba_pipeline"."staging"."stg_players"
)

select
    from_field

from child
left join parent
    on child.from_field = parent.to_field

where parent.to_field is null


