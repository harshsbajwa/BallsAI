#!/usr/bin/env python3
"""Train ML models"""

import sys
import os
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.prediction_model import NBAMLPipeline
from src.models.database import engine
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train ML models"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA is available! Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available. Training on CPU.")

    try:
        pipeline = NBAMLPipeline()

        logger.info("Loading training data...")

        with engine.connect() as connection:
            team_stats_query = """
            SELECT
                ts.*,
                g.home_team_id,
                g.away_team_id,
                g.home_score,
                g.away_score
            FROM team_statistics ts
            JOIN games g ON ts.game_id = g.game_id
            WHERE ts.game_date >= '2020-01-01'
            """
            logger.info("Reading team statistics from database...")
            result = connection.execute(text(team_stats_query))
            team_stats_df = pd.DataFrame(result.fetchall(), columns=result.keys())

            player_stats_query = """
            SELECT ps.*, p.first_name, p.last_name
            FROM player_statistics ps
            JOIN players p ON ps.person_id = p.person_id
            WHERE ps.game_date >= '2020-01-01'
            """
            logger.info("Reading player statistics from database...")
            result = connection.execute(text(player_stats_query))
            player_stats_df = pd.DataFrame(result.fetchall(), columns=result.keys())

        logger.info("Training game prediction model...")
        pipeline.train_game_predictor(team_stats_df)

        logger.info("Training player projection model...")
        pipeline.train_player_projector(player_stats_df)

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
