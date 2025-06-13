from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_daily_analytics(spark: SparkSession, execution_date: str):
    """Run daily analytics computations using PySpark"""
    
    db_properties = {
        "user": "postgres",
        "password": "password",
        "driver": "org.postgresql.Driver",
        "url": "jdbc:postgresql://postgres-service:5432/nba_pipeline"
    }
    
    try:
        player_stats_df = spark.read.jdbc(
            url=db_properties["url"],
            table="player_statistics",
            properties=db_properties
        )
        
        team_stats_df = spark.read.jdbc(
            url=db_properties["url"],
            table="team_statistics",
            properties=db_properties
        )
        
        games_df = spark.read.jdbc(
            url=db_properties["url"],
            table="games",
            properties=db_properties
        )
        
        player_stats_df.createOrReplaceTempView("player_stats")
        team_stats_df.createOrReplaceTempView("team_stats")
        games_df.createOrReplaceTempView("games")
        
        player_momentum_df = calculate_player_momentum(spark)
        team_strength_df = calculate_team_strength(spark)
        head_to_head_df = calculate_head_to_head_records(spark)
        
        save_analytics_results(
            spark, 
            player_momentum_df, 
            team_strength_df, 
            head_to_head_df,
            db_properties
        )
        
        logger.info(f"Successfully completed analytics for {execution_date}")
        
    except Exception as e:
        logger.error(f"Error in analytics job: {e}")
        raise

def calculate_player_momentum(spark: SparkSession):
    """Calculate player momentum metrics"""
    momentum_query = """
    SELECT 
        person_id,
        game_date,
        points,
        assists,
        rebounds_total,
        -- Calculate momentum score based on recent performance trend
        AVG(points) OVER (
            PARTITION BY person_id 
            ORDER BY game_date 
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as points_5_game_avg,
        AVG(points) OVER (
            PARTITION BY person_id 
            ORDER BY game_date 
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) as points_10_game_avg,
        STDDEV(points) OVER (
            PARTITION BY person_id 
            ORDER BY game_date 
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) as points_volatility,
        -- Performance trend indicator
        CASE 
            WHEN AVG(points) OVER (
                PARTITION BY person_id 
                ORDER BY game_date 
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            ) > AVG(points) OVER (
                PARTITION BY person_id 
                ORDER BY game_date 
                ROWS BETWEEN 9 PRECEDING AND 5 PRECEDING
            ) THEN 'IMPROVING'
            ELSE 'DECLINING'
        END as performance_trend
    FROM player_stats
    WHERE game_date >= CURRENT_DATE - INTERVAL '30 days'
    """
    
    return spark.sql(momentum_query)

def calculate_team_strength(spark: SparkSession):
    """Calculate team strength metrics"""
    strength_query = """
    SELECT 
        team_id,
        COUNT(*) as games_played,
        AVG(team_score) as avg_points_scored,
        AVG(opponent_score) as avg_points_allowed,
        AVG(team_score - opponent_score) as avg_margin,
        STDDEV(team_score - opponent_score) as margin_consistency,
        SUM(CASE WHEN win THEN 1 ELSE 0 END) / COUNT(*) as win_rate,
        -- Strength of schedule (avg opponent win rate)
        AVG(opp_win_rate) as strength_of_schedule,
        -- Recent form (last 10 games)
        AVG(CASE WHEN rn <= 10 AND win THEN 1.0 ELSE 0.0 END) as recent_win_rate
    FROM (
        SELECT 
            ts.*,
            ROW_NUMBER() OVER (PARTITION BY ts.team_id ORDER BY ts.game_date DESC) as rn,
            -- Calculate opponent win rate
            opp_stats.win_rate as opp_win_rate
        FROM team_statistics ts
        LEFT JOIN (
            SELECT 
                team_id,
                SUM(CASE WHEN win THEN 1 ELSE 0 END) / COUNT(*) as win_rate
            FROM team_statistics
            GROUP BY team_id
        ) opp_stats ON ts.opponent_team_id = opp_stats.team_id
        WHERE ts.game_date >= CURRENT_DATE - INTERVAL '60 days'
    ) ranked_games
    GROUP BY team_id
    """
    
    return spark.sql(strength_query)

def calculate_head_to_head_records(spark: SparkSession):
    """Calculate head-to-head records between teams"""
    h2h_query = """
    SELECT 
        LEAST(team_id, opponent_team_id) as team_1,
        GREATEST(team_id, opponent_team_id) as team_2,
        SUM(CASE WHEN team_id < opponent_team_id AND win THEN 1
                 WHEN team_id > opponent_team_id AND NOT win THEN 1
                 ELSE 0 END) as team_1_wins,
        SUM(CASE WHEN team_id > opponent_team_id AND win THEN 1
                 WHEN team_id < opponent_team_id AND NOT win THEN 1
                 ELSE 0 END) as team_2_wins,
        COUNT(*) as total_games,
        AVG(ABS(team_score - opponent_score)) as avg_margin,
        MAX(game_date) as last_meeting
    FROM team_statistics
    WHERE game_date >= CURRENT_DATE - INTERVAL '365 days'
    GROUP BY LEAST(team_id, opponent_team_id), GREATEST(team_id, opponent_team_id)
    HAVING COUNT(*) > 0
    """
    
    return spark.sql(h2h_query)

def save_analytics_results(spark, player_momentum_df, team_strength_df, head_to_head_df, db_properties):
    """Save analytics results back to PostgreSQL"""
    
    player_momentum_df.write \
        .mode("overwrite") \
        .jdbc(
            url=db_properties["url"],
            table="analytics_player_momentum",
            properties=db_properties
        )
    
    team_strength_df.write \
        .mode("overwrite") \
        .jdbc(
            url=db_properties["url"],
            table="analytics_team_strength",
            properties=db_properties
        )
    
    head_to_head_df.write \
        .mode("overwrite") \
        .jdbc(
            url=db_properties["url"],
            table="analytics_head_to_head",
            properties=db_properties
        )
