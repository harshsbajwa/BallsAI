from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
import logging
import argparse
from datetime import datetime
from config.settings import settings


logger = logging.getLogger(__name__)

class AdvancedNBAFeatureEngineer:
    """Feature engineering with advanced metrics implementation"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.spark = SparkSession.builder \
            .appName("NBA Advanced Feature Engineering") \
            .config("spark.jars.packages", settings.SPARK_BIGQUERY_CONNECTOR) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Initialized Advanced NBA Feature Engineer")

    def load_data_with_filter(self, table_name: str, date_filter: str = None):
        """Load data with optional date filtering for incremental processing"""
        try:
            df = self.spark.read \
                .format("bigquery") \
                .option("table", f"{self.project_id}.{settings.DATASET_ID}.{table_name}") \
                .load()
            
            if date_filter:
                df = df.filter(col("game_date") >= date_filter)
                logger.info(f"Applied date filter: {date_filter}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {table_name}: {e}")
            raise

    def calculate_advanced_player_metrics(self, traditional_df, advanced_df):
        """Calculate advanced player metrics including PER, TS%, ORtg/DRtg"""
        
        # Join traditional and advanced stats
        player_combined = traditional_df.alias("trad").join(
            advanced_df.alias("adv"),
            ["game_id", "player_id", "team_id"],
            "inner"
        ).select(
            col("trad.game_id"),
            col("trad.player_id"),
            col("trad.team_id"),
            col("trad.player_name"),
            col("trad.min"),
            col("trad.pts"),
            col("trad.reb"),
            col("trad.ast"),
            col("trad.stl"),
            col("trad.blk"),
            col("trad.turnover"),
            col("trad.fgm"),
            col("trad.fga"),
            col("trad.fg3m"),
            col("trad.fg3a"),
            col("trad.ftm"),
            col("trad.fta"),
            col("trad.oreb"),
            col("trad.dreb"),
            col("trad.pf"),
            col("adv.off_rating"),
            col("adv.def_rating"),
            col("adv.net_rating"),
            col("adv.ts_pct"),
            col("adv.usg_pct"),
            col("adv.pace"),
            col("adv.efg_pct"),
            col("adv.ast_pct"),
            col("adv.reb_pct")
        )
        
        # Convert minutes to decimal for calculations
        player_combined = player_combined.withColumn(
            "minutes_decimal",
            when(col("min").isNotNull() & (col("min") != ""),
                 regexp_extract(col("min"), r"(\d+):(\d+)", 1).cast("float") +
                 (regexp_extract(col("min"), r"(\d+):(\d+)", 2).cast("float") / 60)
            ).otherwise(0.0)
        )
        
        # Calculate True Shooting Percentage (manual calculation for validation)
        player_combined = player_combined.withColumn(
            "true_shooting_attempts",
            col("fga") + (lit(0.44) * col("fta"))
        ).withColumn(
            "ts_pct_calculated",
            when(col("true_shooting_attempts") > 0,
                 col("pts") / (lit(2) * col("true_shooting_attempts"))
            ).otherwise(0.0)
        )
        
        # Calculate Effective Field Goal Percentage
        player_combined = player_combined.withColumn(
            "efg_pct_calculated",
            when(col("fga") > 0,
                 (col("fgm") + lit(0.5) * col("fg3m")) / col("fga")
            ).otherwise(0.0)
        )
        
        # Player Efficiency Rating (Simplified version)
        # Note: Full PER calculation requires league averages - this is a simplified version
        player_combined = player_combined.withColumn(
            "per_simplified",
            when(col("minutes_decimal") > 0,
                 (col("pts") + col("reb") + col("ast") + col("stl") + col("blk") 
                  - col("turnover") - (col("fga") - col("fgm")) 
                  - (col("fta") - col("ftm"))) / col("minutes_decimal") * 15
            ).otherwise(0.0)
        )
        
        # Game Score (More practical than full PER)
        player_combined = player_combined.withColumn(
            "game_score",
            col("pts") + lit(0.4) * col("fgm") - lit(0.7) * col("fga") 
            - lit(0.4) * (col("fta") - col("ftm")) + lit(0.7) * col("oreb") 
            + lit(0.3) * col("dreb") + col("stl") + lit(0.7) * col("ast") 
            + lit(0.7) * col("blk") - lit(0.4) * col("pf") - col("turnover")
        )
        
        return player_combined

    def calculate_team_aggregated_stats(self, player_stats_df):
        """Team aggregation with advanced metrics"""
        
        team_stats = player_stats_df.groupBy("game_id", "team_id").agg(
            # Basic aggregations
            sum("pts").alias("team_points"),
            sum("reb").alias("team_rebounds"),
            sum("ast").alias("team_assists"),
            sum("stl").alias("team_steals"),
            sum("blk").alias("team_blocks"),
            sum("turnover").alias("team_turnovers"),
            sum("fgm").alias("team_fg_made"),
            sum("fga").alias("team_fg_attempted"),
            sum("fg3m").alias("team_3p_made"),
            sum("fg3a").alias("team_3p_attempted"),
            sum("ftm").alias("team_ft_made"),
            sum("fta").alias("team_ft_attempted"),
            sum("oreb").alias("team_oreb"),
            sum("dreb").alias("team_dreb"),
            sum("pf").alias("team_fouls"),
            
            # Advanced aggregations
            avg("off_rating").alias("avg_off_rating"),
            avg("def_rating").alias("avg_def_rating"),
            avg("ts_pct").alias("avg_ts_pct"),
            avg("usg_pct").alias("avg_usg_pct"),
            avg("pace").alias("avg_pace"),
            avg("game_score").alias("avg_game_score"),
            
            # Distribution metrics
            stddev("pts").alias("pts_std"),
            max("pts").alias("top_scorer_pts"),
            count("*").alias("players_played"),
            
            # Advanced team metrics
            sum("minutes_decimal").alias("total_minutes"),
            avg("per_simplified").alias("avg_per")
        )
        
        # Calculate team shooting percentages and advanced metrics
        team_stats = team_stats.withColumn(
            "team_fg_pct",
            when(col("team_fg_attempted") > 0, 
                 col("team_fg_made") / col("team_fg_attempted")).otherwise(0.0)
        ).withColumn(
            "team_3p_pct",
            when(col("team_3p_attempted") > 0,
                 col("team_3p_made") / col("team_3p_attempted")).otherwise(0.0)
        ).withColumn(
            "team_ft_pct",
            when(col("team_ft_attempted") > 0,
                 col("team_ft_made") / col("team_ft_attempted")).otherwise(0.0)
        ).withColumn(
            "team_efg_pct",
            when(col("team_fg_attempted") > 0,
                 (col("team_fg_made") + lit(0.5) * col("team_3p_made")) / col("team_fg_attempted")
            ).otherwise(0.0)
        ).withColumn(
            "team_ts_pct",
            when((col("team_fg_attempted") + lit(0.44) * col("team_ft_attempted")) > 0,
                 col("team_points") / (lit(2) * (col("team_fg_attempted") + lit(0.44) * col("team_ft_attempted")))
            ).otherwise(0.0)
        ).withColumn(
            "assist_turnover_ratio",
            when(col("team_turnovers") > 0,
                 col("team_assists") / col("team_turnovers")
            ).otherwise(col("team_assists"))
        ).withColumn(
            "offensive_rebound_pct",
            when((col("team_oreb") + col("team_dreb")) > 0,
                 col("team_oreb") / (col("team_oreb") + col("team_dreb"))
            ).otherwise(0.0)
        )
        
        # Estimated possessions (Dean Oliver's formula)
        team_stats = team_stats.withColumn(
            "possessions_est",
            lit(0.5) * (
                col("team_fg_attempted") + lit(0.44) * col("team_ft_attempted") 
                - lit(1.07) * col("team_oreb") + col("team_turnovers")
            )
        ).withColumn(
            "offensive_rating_est",
            when(col("possessions_est") > 0,
                 col("team_points") / col("possessions_est") * lit(100)
            ).otherwise(0.0)
        )
        
        return team_stats

    def create_rolling_features(self, games_df, team_stats_df, windows=None):
        """Create rolling features with multiple time windows"""
        
        if windows is None:
            windows = settings.ROLLING_WINDOWS  # [3, 5, 10]
        
        # Join team stats with game dates
        team_stats_with_dates = team_stats_df.join(
            games_df.select("game_id", "game_date"),
            "game_id"
        )
        
        # Create rolling features for each window size
        for window_size in windows:
            team_window = Window.partitionBy("team_id") \
                               .orderBy("game_date") \
                               .rowsBetween(-window_size, -1)  # Exclude current game
            
            # Core stats rolling averages
            stat_columns = [
                "team_points", "team_rebounds", "team_assists", "team_steals",
                "team_blocks", "team_turnovers", "team_fg_pct", "team_3p_pct",
                "team_ft_pct", "team_efg_pct", "team_ts_pct", "assist_turnover_ratio",
                "offensive_rating_est", "avg_off_rating", "avg_def_rating", "avg_pace"
            ]
            
            for stat in stat_columns:
                team_stats_with_dates = team_stats_with_dates.withColumn(
                    f"{stat}_avg_L{window_size}",
                    avg(col(stat)).over(team_window)
                ).withColumn(
                    f"{stat}_std_L{window_size}",
                    stddev(col(stat)).over(team_window)
                ).withColumn(
                    f"{stat}_trend_L{window_size}",
                    # Simple trend: current vs average
                    when(avg(col(stat)).over(team_window) > 0,
                         (col(stat) - avg(col(stat)).over(team_window)) / avg(col(stat)).over(team_window)
                    ).otherwise(0.0)
                )
        
        return team_stats_with_dates

    def create_advanced_matchup_features(self, games_df, team_stats_df):
        """Create sophisticated matchup features"""
        
        # Get completed games for historical analysis
        completed_games = games_df.filter(col("wl_home").isNotNull())
        
        # Create team pair identifier (consistent ordering)
        h2h_games = completed_games.withColumn(
            "team_pair",
            concat_ws("-",
                     least(col("home_team_id"), col("away_team_id")),
                     greatest(col("home_team_id"), col("away_team_id")))
        ).withColumn(
            "home_is_team1",
            col("home_team_id") < col("away_team_id")
        )
        
        # Calculate head-to-head statistics
        h2h_stats = h2h_games.withColumn(
            "team1_score",
            when(col("home_is_team1"), col("home_team_score")).otherwise(col("away_team_score"))
        ).withColumn(
            "team2_score", 
            when(col("home_is_team1"), col("away_team_score")).otherwise(col("home_team_score"))
        ).withColumn(
            "team1_win",
            col("team1_score") > col("team2_score")
        ).withColumn(
            "score_margin",
            col("team1_score") - col("team2_score")
        )
        
        # Aggregate H2H stats by team pair
        h2h_aggregated = h2h_stats.groupBy("team_pair").agg(
            count("*").alias("h2h_games_played"),
            avg(col("team1_win").cast("int")).alias("team1_h2h_win_pct"),
            avg("score_margin").alias("avg_margin_team1"),
            stddev("score_margin").alias("margin_volatility"),
            avg("team1_score").alias("avg_score_team1"),
            avg("team2_score").alias("avg_score_team2"),
            max("game_date").alias("last_h2h_date")
        )
        
        # Join back to all games (including future games)
        games_with_h2h = games_df.withColumn(
            "team_pair",
            concat_ws("-",
                     least(col("home_team_id"), col("away_team_id")),
                     greatest(col("home_team_id"), col("away_team_id")))
        ).withColumn(
            "home_is_team1",
            col("home_team_id") < col("away_team_id")
        ).join(h2h_aggregated, "team_pair", "left")
        
        # Calculate home team perspective features
        games_with_h2h = games_with_h2h.withColumn(
            "home_h2h_win_pct",
            when(col("home_is_team1"), col("team1_h2h_win_pct"))
            .otherwise(1 - col("team1_h2h_win_pct"))
        ).withColumn(
            "home_h2h_avg_margin",
            when(col("home_is_team1"), col("avg_margin_team1"))
            .otherwise(-col("avg_margin_team1"))
        ).withColumn(
            "days_since_last_h2h",
            when(col("last_h2h_date").isNotNull(),
                 datediff(col("game_date"), col("last_h2h_date")))
            .otherwise(999)  # Large number for teams that haven't played recently
        )
        
        return games_with_h2h.drop("team_pair", "home_is_team1", "team1_h2h_win_pct", "avg_margin_team1")

    def create_rest_and_schedule_features(self, games_df):
        """Rest and schedule analysis"""
        
        # Rest calculations for both teams
        home_window = Window.partitionBy("home_team_id").orderBy("game_date")
        away_window = Window.partitionBy("away_team_id").orderBy("game_date")
        
        games_with_rest = games_df.withColumn(
            "home_prev_game_date",
            lag("game_date", 1).over(home_window)
        ).withColumn(
            "away_prev_game_date",
            lag("game_date", 1).over(away_window)
        ).withColumn(
            "home_next_game_date",
            lead("game_date", 1).over(home_window)
        ).withColumn(
            "away_next_game_date", 
            lead("game_date", 1).over(away_window)
        )
        
        # Calculate rest days
        games_with_rest = games_with_rest.withColumn(
            "home_rest_days",
            when(col("home_prev_game_date").isNotNull(),
                 datediff("game_date", "home_prev_game_date")).otherwise(3)
        ).withColumn(
            "away_rest_days",
            when(col("away_prev_game_date").isNotNull(),
                 datediff("game_date", "away_prev_game_date")).otherwise(3)
        ).withColumn(
            "home_days_until_next",
            when(col("home_next_game_date").isNotNull(),
                 datediff("home_next_game_date", "game_date")).otherwise(3)
        ).withColumn(
            "away_days_until_next",
            when(col("away_next_game_date").isNotNull(),
                 datediff("away_next_game_date", "game_date")).otherwise(3)
        )
        
        # Rest advantage and situational features
        games_with_rest = games_with_rest.withColumn(
            "rest_advantage",
            col("home_rest_days") - col("away_rest_days")
        ).withColumn(
            "home_back_to_back",
            (col("home_rest_days") == 1).cast("int")
        ).withColumn(
            "away_back_to_back",
            (col("away_rest_days") == 1).cast("int")
        ).withColumn(
            "home_schedule_compressed",
            (col("home_rest_days") <= 1).and(col("home_days_until_next") <= 1).cast("int")
        ).withColumn(
            "away_schedule_compressed",
            (col("away_rest_days") <= 1).and(col("away_days_until_next") <= 1).cast("int")
        ).withColumn(
            "both_teams_rested",
            (col("home_rest_days") >= 2).and(col("away_rest_days") >= 2).cast("int")
        ).withColumn(
            "rest_mismatch",
            abs(col("home_rest_days") - col("away_rest_days"))
        )
        
        return games_with_rest.drop(
            "home_prev_game_date", "away_prev_game_date", 
            "home_next_game_date", "away_next_game_date"
        )

    def create_time_and_context_features(self, games_df):
        """Temporal and contextual features"""
        
        games = games_df.withColumn(
            "day_of_week", dayofweek("game_date")
        ).withColumn(
            "month", month("game_date")
        ).withColumn(
            "day_of_year", dayofyear("game_date")
        ).withColumn(
            "is_weekend",
            col("day_of_week").isin([1, 7]).cast("int")  # Sunday=1, Saturday=7
        ).withColumn(
            "is_weekday",
            col("day_of_week").isin([2, 3, 4, 5, 6]).cast("int")
        ).withColumn(
            "season_stage",
            when(col("month").isin([10, 11]), "early_season")
            .when(col("month").isin([12, 1, 2]), "mid_season")
            .when(col("month").isin([3, 4]), "late_season")
            .otherwise("off_season")
        ).withColumn(
            "season_progress",
            # Approximate season progress (Oct 1 = day 274)
            when(col("day_of_year") >= 274, (col("day_of_year") - 274) / 191.0)  # Oct-Apr
            .when(col("day_of_year") <= 120, (col("day_of_year") + 91) / 191.0)   # Jan-Apr of next year
            .otherwise(0.0)
        )
        
        return games

    def create_final_feature_set(self, incremental_date: str = None):
        """Create comprehensive feature set with all advanced metrics"""
        
        logger.info("Starting advanced feature engineering pipeline...")
        start_time = datetime.now()
        
        try:
            # Load base data with optional incremental filtering
            logger.info("Loading base data...")
            games_df = self.load_data_with_filter("raw_games", incremental_date)
            player_traditional_df = self.load_data_with_filter("raw_player_boxscores_traditional", incremental_date)
            player_advanced_df = self.load_data_with_filter("raw_player_boxscores_advanced", incremental_date)
            
            # Calculate advanced player metrics
            logger.info("Calculating advanced player metrics...")
            player = self.calculate_advanced_player_metrics(
                player_traditional_df, player_advanced_df
            )
            
            # Create team aggregated statistics
            logger.info("Creating team aggregated statistics...")
            team_stats = self.calculate_team_aggregated_stats(player)
            
            # Create rolling features
            logger.info("Creating rolling team features...")
            team_rolling = self.create_rolling_features(games_df, team_stats)
            
            # Create advanced matchup features
            logger.info("Creating advanced matchup features...")
            games_with_matchup = self.create_advanced_matchup_features(games_df, team_stats)
            
            # Create rest and schedule features
            logger.info("Creating rest and schedule features...")
            games_with_rest = self.create_rest_and_schedule_features(games_with_matchup)
            
            # Create time and context features
            logger.info("Creating time and context features...")
            games = self.create_time_and_context_features(games_with_rest)
            
            # Join team features for both home and away teams
            logger.info("Joining team features...")
            
            # Home team features
            home_features = team_rolling.select(
                col("game_id"),
                col("team_id").alias("home_team_id"),
                *[col(c).alias(f"home_{c}") for c in team_rolling.columns 
                  if c not in ["game_id", "team_id", "game_date"]]
            )
            
            # Away team features
            away_features = team_rolling.select(
                col("game_id"),
                col("team_id").alias("away_team_id"),
                *[col(c).alias(f"away_{c}") for c in team_rolling.columns 
                  if c not in ["game_id", "team_id", "game_date"]]
            )
            
            # Final comprehensive join
            final_df = games \
                .join(home_features, ["game_id", "home_team_id"], "left") \
                .join(away_features, ["game_id", "away_team_id"], "left")
            
            # Create differential features (home vs away)
            differential_features = [
                "team_points_avg_L5", "team_rebounds_avg_L5", "team_assists_avg_L5",
                "team_fg_pct_avg_L5", "team_3p_pct_avg_L5", "team_ts_pct_avg_L5",
                "offensive_rating_est_avg_L5", "avg_off_rating_avg_L5", "avg_def_rating_avg_L5",
                "avg_pace_avg_L5"
            ]
            
            for feature in differential_features:
                home_col = f"home_{feature}"
                away_col = f"away_{feature}"
                diff_col = f"diff_{feature}"
                
                if home_col in final_df.columns and away_col in final_df.columns:
                    final_df = final_df.withColumn(
                        diff_col,
                        when((col(home_col).isNotNull()) & (col(away_col).isNotNull()),
                             col(home_col) - col(away_col)).otherwise(0.0)
                    )
            
            # Fill nulls with appropriate defaults
            logger.info("Cleaning and finalizing features...")
            
            # Get all feature columns (excluding identifiers and targets)
            feature_columns = [col for col in final_df.columns 
                             if col.startswith(("home_", "away_", "diff_", "rest_", "h2h_", 
                                               "is_", "day_", "month", "season_"))]
            
            # Fill nulls with 0 for most features
            for col_name in feature_columns:
                final_df = final_df.fillna(0.0, subset=[col_name])
            
            # Add metadata
            final_df = final_df.withColumn(
                "features_created_at", 
                current_timestamp()
            ).withColumn(
                "feature_engineering_version",
                lit("advanced_v1.0")
            )
            
            # Log feature engineering summary
            feature_count = len([col for col in final_df.columns if col.startswith(("home_", "away_", "diff_"))])
            processing_time = datetime.now() - start_time
            
            logger.info(f"Advanced feature engineering completed!")
            logger.info(f"  - Total features: {feature_count}")
            logger.info(f"  - Processing time: {processing_time}")
            logger.info(f"  - Records processed: {final_df.count()}")
            
            return final_df
            
        except Exception as e:
            logger.error(f"Error in advanced feature engineering: {e}")
            raise

    def save_features_to_bigquery(self, df, table_name="feature_set", mode="overwrite"):
        """Save features with optimized BigQuery settings"""
        try:
            # Write with optimized settings
            writer = df.coalesce(10)  # Reduce number of output files
            
            writer.write \
                .format("bigquery") \
                .option("table", f"{self.project_id}.{settings.DATASET_ID}.{table_name}") \
                .option("writeMethod", "direct") \
                .option("createDisposition", "CREATE_IF_NEEDED") \
                .mode(mode) \
                .save()
            
            logger.info(f"Features successfully saved to {table_name}")
            
        except Exception as e:
            logger.error(f"Error saving features to BigQuery: {e}")
            raise

    def close(self):
        """Clean up Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session closed")

def main():
    """Main function for running feature engineering as standalone script"""
    parser = argparse.ArgumentParser(description="NBA Advanced Feature Engineering")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--incremental-date", help="Date for incremental processing (YYYY-MM-DD)")
    parser.add_argument("--output-table", default="feature_set", help="Output table name")
    parser.add_argument("--mode", default="overwrite", choices=["overwrite", "append"], help="Write mode")
    
    args = parser.parse_args()
    
    engineer = AdvancedNBAFeatureEngineer(args.project_id)
    
    try:
        # Create feature set
        feature_df = engineer.create_final_feature_set(args.incremental_date)
        
        # Save to BigQuery
        engineer.save_features_to_bigquery(feature_df, args.output_table, args.mode)
        
        logger.info("Feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise
    finally:
        engineer.close()

if __name__ == "__main__":
    main()