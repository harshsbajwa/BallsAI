from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import col, avg, stddev, when, lit, coalesce, sum
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_player_impact_rating_job(spark: SparkSession):
	"""
    Calculates a Player Impact Rating (PIR) using Spark.
    """
	postgres_host = os.getenv("POSTGRES_HOST", "postgres-service")
	db_properties = {
		"user": os.getenv("DBT_USER", "postgres"),
		"password": os.getenv("DBT_PASSWORD", "password"),
		"driver": "org.postgresql.Driver",
	}
	db_url = f"jdbc:postgresql://{postgres_host}:5432/nba_pipeline"

	# Read from clean dbt staging models
	logger.info("Reading data from dbt staging models...")
	player_stats_df = spark.read.jdbc(
		url=db_url,
		table="staging.stg_player_statistics",
		properties=db_properties,
	)

	# Filter for games with minutes played to avoid division by zero
	player_stats_df = player_stats_df.filter(col("num_minutes") > 0)

	# Calculate True Shooting Percentage (TS%)
	player_stats_df = player_stats_df.withColumn(
		"ts_percentage",
		when(
			(
				col("field_goals_attempted")
				+ 0.44 * col("free_throws_attempted")
			)
			> 0,
			col("points")
			/ (
				2
				* (
					col("field_goals_attempted")
					+ 0.44 * col("free_throws_attempted")
				)
			),
		).otherwise(0),
	)

	# Calculate game-level impact score
	game_impact_score = (
		col("points")
		+ (col("rebounds_total") * 0.8)
		+ (col("assists") * 1.2)
		+ (col("steals") * 1.5)
		+ (col("blocks") * 1.1)
		- (col("field_goals_attempted") - col("field_goals_made"))
		- (col("free_throws_attempted") - col("free_throws_made")) * 0.5
		- (col("turnovers") * 1.5)
	)

	player_stats_df = player_stats_df.withColumn(
		"game_impact_score_per_min", game_impact_score / col("num_minutes")
	)

	# Aggregate to Season Level
	logger.info("Aggregating stats to the season level...")
	season_stats_df = player_stats_df.groupBy("person_id", "season_year").agg(
		avg("game_impact_score_per_min").alias("avg_impact_per_min"),
		avg("ts_percentage").alias("avg_ts_percentage"),
		avg("plus_minus_points").alias("avg_plus_minus"),
		(sum(col("points")) / sum(col("num_minutes")) * 36).alias(
			"points_per_36"
		),
		(sum(col("assists")) / sum(col("num_minutes")) * 36).alias(
			"assists_per_36"
		),
		(sum(col("rebounds_total")) / sum(col("num_minutes")) * 36).alias(
			"rebounds_per_36"
		),
	)

	# Standardize Metrics (Z-Scores) for Comparability
	logger.info("Standardizing metrics using Z-scores...")
	season_window = Window.partitionBy("season_year")

	# Calculate mean and stddev for each metric over the season
	metrics_to_standardize = [
		"avg_impact_per_min",
		"avg_ts_percentage",
		"avg_plus_minus",
	]
	for metric in metrics_to_standardize:
		mean_val = avg(col(metric)).over(season_window)
		stddev_val = stddev(col(metric)).over(season_window)
		# FIX: Handle division by zero when standard deviation is 0
		season_stats_df = season_stats_df.withColumn(
			f"{metric}_z",
			when(stddev_val == 0, lit(0.0)).otherwise(
				(col(metric) - mean_val) / stddev_val
			),
		)

	# Fill any null Z-scores (from single-player seasons) with 0
	for metric in metrics_to_standardize:
		season_stats_df = season_stats_df.withColumn(
			f"{metric}_z", coalesce(col(f"{metric}_z"), lit(0.0))
		)

	# Combine the standardized scores into a final weighted rating
	logger.info("Calculating final Player Impact Rating (PIR)...")
	pir_df = season_stats_df.withColumn(
		"player_impact_rating",
		(
			col("avg_impact_per_min_z") * 0.50
			+ col("avg_ts_percentage_z")
			* 0.30  # 30% weight on scoring efficiency
			+ col("avg_plus_minus_z") * 0.20
		),  # 20% weight on on-court team impact
	).select(
		"person_id",
		"season_year",
		"player_impact_rating",
		"points_per_36",
		"rebounds_per_36",
		"assists_per_36",
		"avg_ts_percentage",
	)

	# Save Results
	logger.info("Saving PIR results to the database...")
	pir_df.write.mode("overwrite").jdbc(
		url=db_url,
		table="analytics.analytics_player_impact_rating",
		properties=db_properties,
	)
	logger.info("Successfully completed Player Impact Rating job.")