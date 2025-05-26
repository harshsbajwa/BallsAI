from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
import logging
from google.cloud import bigquery
from config.settings import settings


logger = logging.getLogger(__name__)

class NBAFeatureEngineer:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.spark = SparkSession.builder \
            .appName("NBA Feature Engineering") \
            .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")

    def load_data_from_bigquery(self, table_name: str):
        """Load data from BigQuery with error handling"""
        try:
            return self.spark.read \
                .format("bigquery") \
                .option("table", f"{self.project_id}.{settings.DATASET_ID}.{table_name}") \
                .load()
        except Exception as e:
            logger.error(f"Error loading {table_name}: {e}")
            raise


    def create_team_aggregated_stats(self, player_stats_df):
        """Create team-level aggregated statistics from player stats"""
        team_stats = player_stats_df.groupBy("game_id", "team_id").agg(
            # Basic stats
            sum("points").alias("team_points"),
            sum("rebounds").alias("team_rebounds"),
            sum("assists").alias("team_assists"),
            sum("steals").alias("team_steals"),
            sum("blocks").alias("team_blocks"),
            sum("turnovers").alias("team_turnovers"),
            
            # Shooting stats
            sum("field_goals_made").alias("team_fg_made"),
            sum("field_goals_attempted").alias("team_fg_attempted"),
            sum("three_pointers_made").alias("team_3p_made"),
            sum("three_pointers_attempted").alias("team_3p_attempted"),
            sum("free_throws_made").alias("team_ft_made"),
            sum("free_throws_attempted").alias("team_ft_attempted"),
            
            # Advanced stats
            count("*").alias("players_played"),
            avg("plus_minus").alias("avg_plus_minus"),
            stddev("points").alias("points_distribution"),
            max("points").alias("top_scorer_points")
        )

        # Calculate percentages and advanced metrics
        team_stats = team_stats.withColumn(
            "team_fg_pct",
            when(col("team_fg_attempted") > 0, 
                 col("team_fg_made") / col("team_fg_attempted")).otherwise(0)
        ).withColumn(
            "team_3p_pct",
            when(col("team_3p_attempted") > 0, 
                 col("team_3p_made") / col("team_3p_attempted")).otherwise(0)
        ).withColumn(
            "team_ft_pct",
            when(col("team_ft_attempted") > 0, 
                 col("team_ft_made") / col("team_ft_attempted")).otherwise(0)
        ).withColumn(
            "effective_fg_pct",
            when(col("team_fg_attempted") > 0,
                 (col("team_fg_made") + 0.5 * col("team_3p_made")) / col("team_fg_attempted")).otherwise(0)
        ).withColumn(
            "true_shooting_pct",
            when((col("team_fg_attempted") + 0.44 * col("team_ft_attempted")) > 0,
                 col("team_points") / (2 * (col("team_fg_attempted") + 0.44 * col("team_ft_attempted")))).otherwise(0)
        ).withColumn(
            "assist_turnover_ratio",
            when(col("team_turnovers") > 0, 
                 col("team_assists") / col("team_turnovers")).otherwise(col("team_assists"))
        )

        return team_stats


    def create_rolling_team_features(self, games_df, team_stats_df, windows=[3, 5, 10]):
        """Create rolling averages for team performance"""
        
        # Join games with team stats to get dates
        team_stats_with_dates = team_stats_df.join(
            games_df.select("game_id", "game_date"), 
            "game_id"
        )
        
        # Create rolling features for each window
        for window_size in windows:
            team_window = Window.partitionBy("team_id") \
                               .orderBy("game_date") \
                               .rowsBetween(-window_size + 1, 0)
            
            stat_columns = [
                "team_points", "team_rebounds", "team_assists", "team_steals",
                "team_blocks", "team_turnovers", "team_fg_pct", "team_3p_pct",
                "effective_fg_pct", "true_shooting_pct", "assist_turnover_ratio"
            ]
            
            for stat in stat_columns:
                team_stats_with_dates = team_stats_with_dates.withColumn(
                    f"{stat}_avg_{window_size}",
                    avg(col(stat)).over(team_window)
                ).withColumn(
                    f"{stat}_std_{window_size}",
                    stddev(col(stat)).over(team_window)
                )
        
        return team_stats_with_dates


    def create_head_to_head_features(self, games_df):
        """Create head-to-head matchup features"""
        completed_games = games_df.filter(col("home_win").isNotNull())
        
        # Create matchup identifier
        h2h_stats = completed_games.withColumn(
            "team_pair",
            concat_ws("-", 
                     least(col("home_team_id"), col("away_team_id")),
                     greatest(col("home_team_id"), col("away_team_id")))
        ).withColumn(
            "home_is_team1",
            col("home_team_id") < col("away_team_id")
        ).withColumn(
            "team1_win",
            when(col("home_is_team1"), col("home_win"))
            .otherwise(~col("home_win"))
        ).withColumn(
            "margin",
            when(col("home_is_team1"), col("home_score") - col("away_score"))
            .otherwise(col("away_score") - col("home_score"))
        )
        
        # Aggregate H2H stats
        h2h_aggregated = h2h_stats.groupBy("team_pair").agg(
            count("*").alias("h2h_games"),
            avg(col("team1_win").cast("int")).alias("team1_h2h_win_pct"),
            avg("margin").alias("avg_h2h_margin"),
            stddev("margin").alias("h2h_margin_std")
        )
        
        # Join back to games
        games_with_h2h = games_df.withColumn(
            "team_pair",
            concat_ws("-", 
                     least(col("home_team_id"), col("away_team_id")),
                     greatest(col("home_team_id"), col("away_team_id")))
        ).withColumn(
            "home_is_team1",
            col("home_team_id") < col("away_team_id")
        ).join(h2h_aggregated, "team_pair", "left")
        
        # Calculate home team H2H features
        games_with_h2h = games_with_h2h.withColumn(
            "home_h2h_win_pct",
            when(col("home_is_team1"), col("team1_h2h_win_pct"))
            .otherwise(1 - col("team1_h2h_win_pct"))
        ).withColumn(
            "home_h2h_margin_avg",
            when(col("home_is_team1"), col("avg_h2h_margin"))
            .otherwise(-col("avg_h2h_margin"))
        )
        
        return games_with_h2h.drop("team_pair", "home_is_team1", "team1_h2h_win_pct", "avg_h2h_margin")


    def create_rest_and_schedule_features(self, games_df):
        """Create rest days and schedule strength features"""
        
        # Rest days calculation
        home_window = Window.partitionBy("home_team_id").orderBy("game_date")
        away_window = Window.partitionBy("away_team_id").orderBy("game_date")
        
        games_with_rest = games_df.withColumn(
            "home_prev_game",
            lag("game_date").over(home_window)
        ).withColumn(
            "away_prev_game", 
            lag("game_date").over(away_window)
        ).withColumn(
            "home_rest_days",
            when(col("home_prev_game").isNotNull(),
                 datediff("game_date", "home_prev_game")).otherwise(3)
        ).withColumn(
            "away_rest_days",
            when(col("away_prev_game").isNotNull(),
                 datediff("game_date", "away_prev_game")).otherwise(3)
        ).withColumn(
            "rest_advantage",
            col("home_rest_days") - col("away_rest_days")
        ).withColumn(
            "home_back_to_back",
            (col("home_rest_days") == 1).cast("int")
        ).withColumn(
            "away_back_to_back", 
            (col("away_rest_days") == 1).cast("int")
        )
        
        return games_with_rest.drop("home_prev_game", "away_prev_game")


    def create_final_feature_set(self):
        """Create the complete feature set for modeling"""
        logger.info("Starting feature engineering pipeline...")
        
        try:
            # Load base data
            logger.info("Loading data from BigQuery...")
            games_df = self.load_data_from_bigquery("game_stats")
            player_stats_df = self.load_data_from_bigquery("player_stats")
            
            # Create team aggregated stats
            logger.info("Creating team aggregated statistics...")
            team_stats = self.create_team_aggregated_stats(player_stats_df)
            
            # Create rolling features
            logger.info("Creating rolling team features...")
            team_rolling = self.create_rolling_team_features(games_df, team_stats)
            
            # Create H2H features
            logger.info("Creating head-to-head features...")
            games_with_h2h = self.create_head_to_head_features(games_df)
            
            # Create rest and schedule features
            logger.info("Creating rest and schedule features...")
            games_with_rest = self.create_rest_and_schedule_features(games_with_h2h)
            
            # Add time-based features
            logger.info("Adding time-based features...")
            games_enhanced = games_with_rest.withColumn(
                "day_of_week", dayofweek("game_date")
            ).withColumn(
                "month", month("game_date")
            ).withColumn(
                "is_weekend", 
                col("day_of_week").isin([1, 7]).cast("int")
            ).withColumn(
                "season_progress",
                (dayofyear("game_date") - 274) / 365.0  # NBA season starts ~Oct 1
            )
            
            # Join team features for home and away teams
            logger.info("Joining team features...")
            home_features = team_rolling.select(
                col("game_id"),
                col("team_id").alias("home_team_id"),
                *[col(c).alias(f"home_{c}") for c in team_rolling.columns 
                  if c not in ["game_id", "team_id", "game_date"]]
            )
            
            away_features = team_rolling.select(
                col("game_id"),
                col("team_id").alias("away_team_id"),
                *[col(c).alias(f"away_{c}") for c in team_rolling.columns 
                  if c not in ["game_id", "team_id", "game_date"]]
            )
            
            # Final join
            final_df = games_enhanced \
                .join(home_features, ["game_id", "home_team_id"], "left") \
                .join(away_features, ["game_id", "away_team_id"], "left")
            
            # Fill nulls and add metadata
            feature_columns = [col for col in final_df.columns 
                             if col.startswith(("home_", "away_", "rest_", "h2h_", "is_", "day_", "month", "season_"))]
            
            for col_name in feature_columns:
                final_df = final_df.fillna(0, subset=[col_name])
            
            final_df = final_df.withColumn("features_created_at", current_timestamp())
            
            logger.info(f"Feature engineering completed. Final dataset has {len(final_df.columns)} columns")
            return final_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise


    def save_features_to_bigquery(self, df, table_name="feature_set"):
        """Save features to BigQuery"""
        try:
            df.write \
                .format("bigquery") \
                .option("table", f"{self.project_id}.{settings.DATASET_ID}.{table_name}") \
                .option("writeMethod", "direct") \
                .mode("overwrite") \
                .save()
            
            logger.info(f"Features saved to {table_name}")
            
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise


    def close(self):
        """Clean up Spark session"""
        if self.spark:
            self.spark.stop()
