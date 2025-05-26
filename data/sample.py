"""
NBA Data Collection Script for 2023-24 Season
"""

import os
import sys
from pathlib import Path
from google.cloud import bigquery
from datetime import datetime
import logging
import time

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from data.collector import NBADataCollector
from config.settings import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to collect 2023-24 NBA season data"""
    
    # Initialize BigQuery client
    try:
        client = bigquery.Client(project=settings.PROJECT_ID)
        logger.info(f"Connected to BigQuery project: {settings.PROJECT_ID}")
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {e}")
        return False
    
    # Initialize NBA Data Collector
    collector = NBADataCollector(client)
    
    # Define 2023-24 season parameters
    season = "2023-24"

    logger.info(f"Starting data collection for {season} season")

    try:
        # Step 1: Collect historical games for the season
        logger.info("Step 1: Collecting historical games...")
        games = collector.collect_historical_games(season_str=season)

        if not games:
            logger.error("No games found for the specified season")
            return False

        logger.info(f"Found {len(games)} games for {season} season")
        
        # Step 2: Upload games to BigQuery
        logger.info("Step 2: Uploading games to BigQuery...")
        collector.upload_to_bigquery("game_stats", games)
        logger.info("Successfully uploaded game stats")
        
        # Step 3: Collect box scores for completed games
        logger.info("Step 3: Collecting player statistics...")
        completed_game_ids = [
            game['game_id'] for game in games 
            if game.get('game_status_id') == 3  # Final games only
        ]
        
        if not completed_game_ids:
            logger.warning("No completed games found")
            return True
            
        logger.info(f"Found {len(completed_game_ids)} completed games")
        
        # Process games in batches to avoid overwhelming the API
        batch_size = 25  # Smaller batch size for reliability
        total_player_stats = []
        
        for i in range(0, len(completed_game_ids), batch_size):
            batch = completed_game_ids[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(completed_game_ids) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} games)")
            
            try:
                player_stats = collector.collect_box_scores(batch)
                if player_stats:
                    total_player_stats.extend(player_stats)
                    logger.info(f"Collected stats for {len(player_stats)} player performances")
                else:
                    logger.warning(f"No player stats returned for batch {batch_num}")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                continue  # Continue with next batch
            
            # Rate limiting between batches
            if i + batch_size < len(completed_game_ids):
                logger.info("Waiting 10 seconds before next batch...")
                time.sleep(10)
        
        # Step 4: Upload player stats to BigQuery
        if total_player_stats:
            logger.info(f"Step 4: Uploading {len(total_player_stats)} player stat records...")
            collector.upload_to_bigquery("player_stats", total_player_stats)
            logger.info("Successfully uploaded player stats")
        else:
            logger.warning("No player stats to upload")
        
        # Step 5: Try to collect betting odds (may not be available)
        logger.info("Step 5: Attempting to collect current betting odds...")
        try:
            odds = collector.collect_betting_odds()
            if odds:
                collector.upload_to_bigquery("betting_odds", odds)
                logger.info(f"Uploaded {len(odds)} betting odds records")
            else:
                logger.info("No current betting odds available")
        except Exception as e:
            logger.warning(f"Could not collect betting odds: {e}")
        
        logger.info("Data collection completed successfully!")
        logger.info(f"Summary:")
        logger.info(f"  - Games collected: {len(games)}")
        logger.info(f"  - Player stats collected: {len(total_player_stats)}")
        logger.info(f"  - Season: {season}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        return False

def verify_collection():
    """Verify the data collection by querying BigQuery"""
    try:
        client = bigquery.Client(project=settings.PROJECT_ID)
        
        # Check game_stats table
        query = f"""
        SELECT 
            COUNT(*) as total_games,
            MIN(game_date) as earliest_game,
            MAX(game_date) as latest_game,
            COUNT(DISTINCT home_team_id) + COUNT(DISTINCT away_team_id) as unique_teams
        FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.game_stats`
        WHERE season = '2023-24'
        """
        
        results = client.query(query).to_dataframe()
        logger.info("Game Stats Summary:")
        logger.info(f"  Total games: {results.iloc[0]['total_games']}")
        logger.info(f"  Date range: {results.iloc[0]['earliest_game']} to {results.iloc[0]['latest_game']}")
        
        # Check player_stats table
        query = f"""
        SELECT 
            COUNT(*) as total_player_stats,
            COUNT(DISTINCT player_id) as unique_players,
            COUNT(DISTINCT game_id) as games_with_stats
        FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.player_stats`
        """
        
        results = client.query(query).to_dataframe()
        logger.info("Player Stats Summary:")
        logger.info(f"  Total player performances: {results.iloc[0]['total_player_stats']}")
        logger.info(f"  Unique players: {results.iloc[0]['unique_players']}")
        logger.info(f"  Games with stats: {results.iloc[0]['games_with_stats']}")
        
    except Exception as e:
        logger.error(f"Error verifying collection: {e}")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NBA Data Collection Script - 2023-24 Season")
    logger.info("=" * 60)
    
    # Check if service account key exists
    if not os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
        logger.error(f"Service account key not found at: {settings.GOOGLE_APPLICATION_CREDENTIALS}")
        logger.error("Please ensure your service account key is in the correct location")
        sys.exit(1)
    
    start_time = datetime.now()
    success = main()
    end_time = datetime.now()
    
    logger.info("=" * 60)
    if success:
        logger.info("Data collection completed successfully!")
        logger.info(f"Total runtime: {end_time - start_time}")
        logger.info("\nRunning verification...")
        verify_collection()
    else:
        logger.error("Data collection failed!")
        logger.error(f"Runtime: {end_time - start_time}")
    
    logger.info("=" * 60)
