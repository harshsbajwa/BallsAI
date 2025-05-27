import time
import requests
import numpy as np
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
from functools import wraps
from google.cloud import bigquery
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, boxscoreadvancedv2
from nba_api.stats.static import teams
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

def defensive_api_call(max_retries=None, base_delay=None, backoff_factor=None):
    """Defensive API wrapper"""
    if max_retries is None:
        max_retries = settings.NBA_API_MAX_RETRIES
    if base_delay is None:
        base_delay = settings.NBA_API_DELAY
    if backoff_factor is None:
        backoff_factor = settings.NBA_API_BACKOFF_FACTOR
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    # Rate limiting on every call
                    time.sleep(base_delay)
                    result = func(*args, **kwargs)
                    if result is not None:
                        return result
                        
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    if "429" in str(e) or "rate" in str(e).lower():
                        delay = base_delay * (backoff_factor ** attempt) + np.random.uniform(0, 2)
                        logger.warning(f"Rate limited on {func.__name__} attempt {attempt + 1}, waiting {delay:.1f}s")
                        time.sleep(delay)
                    elif "404" in str(e):
                        logger.warning(f"404 error on {func.__name__}: {e}")
                        return None  # Don't retry 404s
                    else:
                        delay = base_delay * (attempt + 1)
                        logger.warning(f"Request error on {func.__name__} attempt {attempt + 1}: {e}")
                        time.sleep(delay)
                
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (attempt + 1)
                    logger.warning(f"Unexpected error on {func.__name__} attempt {attempt + 1}: {e}")
                    time.sleep(delay)
            
            logger.error(f"All {max_retries} attempts failed for {func.__name__}: {last_exception}")
            return None
            
        return wrapper
    return decorator

class NBACollector:
    """NBA data collector following defensive design principles"""
    
    def __init__(self, bigquery_client):
        self.client = bigquery_client
        self.dataset_id = settings.DATASET_ID
        self.failed_operations = []
        self.success_metrics = {
            'games_collected': 0,
            'box_scores_collected': 0,
            'api_calls_made': 0,
            'api_failures': 0
        }
        
        # Initialize team cache
        self.team_cache = {}
        self._initialize_team_cache()
    
    def _initialize_team_cache(self):
        """Initialize team cache for faster lookups"""
        try:
            nba_teams = teams.get_teams()
            self.team_cache = {team['id']: team for team in nba_teams}
            logger.info(f"Cached {len(self.team_cache)} NBA teams")
        except Exception as e:
            logger.error(f"Failed to initialize team cache: {e}")
    
    @defensive_api_call()
    def _fetch_games_for_season(self, season_str: str, season_type: str = 'Regular Season'):
        """Defensively fetch games for a season"""
        self.success_metrics['api_calls_made'] += 1
        
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            season_type_nullable=season_type
        )
        
        games_df = gamefinder.get_data_frames()[0]
        return games_df
    
    @defensive_api_call()
    def _fetch_traditional_box_score(self, game_id: str):
        """Defensively fetch traditional box score"""
        self.success_metrics['api_calls_made'] += 1
        
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        return {
            'player_stats': boxscore.player_stats.get_data_frame(),
            'team_stats': boxscore.team_stats.get_data_frame()
        }
    
    @defensive_api_call()
    def _fetch_advanced_box_score(self, game_id: str):
        """Defensively fetch advanced box score"""
        self.success_metrics['api_calls_made'] += 1
        
        boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        return {
            'player_stats': boxscore.player_stats.get_data_frame(),
            'team_stats': boxscore.team_stats.get_data_frame()
        }
    
    def collect_historical_games(self, season_str: str) -> List[Dict]:
        """Collect historical games with comprehensive error handling"""
        logger.info(f"Starting collection for season {season_str}")
        
        try:
            games_df = self._fetch_games_for_season(season_str)
            if games_df is None or games_df.empty:
                logger.warning(f"No games found for season {season_str}")
                return []
            
            # Process unique games only
            unique_games = games_df.drop_duplicates(subset=['GAME_ID'])
            logger.info(f"Found {len(unique_games)} unique games for {season_str}")
            
            processed_games = []
            
            for idx, game_row in unique_games.iterrows():
                try:
                    # Game processing with data validation
                    game_record = self._process_game_row(games_df, game_row)
                    if game_record:
                        processed_games.append(game_record)
                        self.success_metrics['games_collected'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process game {game_row['GAME_ID']}: {e}")
                    self.failed_operations.append({
                        'type': 'game_processing',
                        'game_id': game_row['GAME_ID'],
                        'error': str(e),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    continue
            
            logger.info(f"Successfully processed {len(processed_games)} games for {season_str}")
            return processed_games
            
        except Exception as e:
            logger.error(f"Critical error in historical collection for {season_str}: {e}")
            return []
    
    def _process_game_row(self, games_df: pd.DataFrame, game_row: pd.Series) -> Optional[Dict]:
        """Process individual game row with validation"""
        try:
            # Find opponent
            opponent_games = games_df[
                (games_df['GAME_ID'] == game_row['GAME_ID']) &
                (games_df['TEAM_ID'] != game_row['TEAM_ID'])
            ]
            
            if opponent_games.empty:
                logger.warning(f"No opponent found for game {game_row['GAME_ID']}")
                return None
            
            opponent_game = opponent_games.iloc[0]
            
            # Determine home/away with validation
            is_home_game = 'vs.' in str(game_row['MATCHUP'])
            
            home_team_id = int(game_row['TEAM_ID']) if is_home_game else int(opponent_game['TEAM_ID'])
            away_team_id = int(opponent_game['TEAM_ID']) if is_home_game else int(game_row['TEAM_ID'])
            
            # Score validation
            home_score = self._safe_int_convert(
                game_row['PTS'] if is_home_game else opponent_game['PTS']
            )
            away_score = self._safe_int_convert(
                opponent_game['PTS'] if is_home_game else game_row['PTS']
            )
            
            # Data quality validation
            if home_score is None or away_score is None:
                logger.warning(f"Invalid scores for game {game_row['GAME_ID']}")
                return None
            
            # Game record with data quality metrics
            game_record = {
                'game_id': str(game_row['GAME_ID']),
                'game_date': pd.to_datetime(game_row['GAME_DATE']).strftime('%Y-%m-%d'),
                'season_id': str(game_row['SEASON_ID']),
                'season_year': season_str,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_abbreviation': self._get_team_abbreviation(home_team_id),
                'away_team_abbreviation': self._get_team_abbreviation(away_team_id),
                'home_team_score': home_score,
                'away_team_score': away_score,
                'wl_home': 'W' if home_score > away_score else 'L',
                'wl_away': 'L' if home_score > away_score else 'W',
                'matchup': f"{self._get_team_abbreviation(home_team_id)} vs. {self._get_team_abbreviation(away_team_id)}",
                'plus_minus_home': float(home_score - away_score),
                'plus_minus_away': float(away_score - home_score),
                'load_timestamp': datetime.now(timezone.utc).isoformat(),
                'data_quality_score': self._calculate_game_quality_score(game_row, opponent_game)
            }
            
            return game_record
            
        except Exception as e:
            logger.error(f"Error processing game row {game_row.get('GAME_ID', 'unknown')}: {e}")
            return None
    
    def _calculate_game_quality_score(self, game_row: pd.Series, opponent_game: pd.Series) -> float:
        """Calculate data quality score (0-1) based on completeness and consistency"""
        score = 1.0
        
        # Check for missing essential data
        essential_fields = ['PTS', 'GAME_DATE', 'MATCHUP', 'TEAM_ID']
        for field in essential_fields:
            if pd.isna(game_row.get(field)) or pd.isna(opponent_game.get(field)):
                score -= 0.2
        
        # Check for data consistency
        if game_row['GAME_ID'] != opponent_game['GAME_ID']:
            score -= 0.3
        
        return max(0.0, score)
    
    def collect_box_scores_batch(self, game_ids: List[str]) -> List[Dict]:
        """Collect box scores in batches with comprehensive error handling"""
        all_player_stats = []
        successful_games = 0
        
        for i, game_id in enumerate(game_ids):
            try:
                logger.debug(f"Processing box score {i+1}/{len(game_ids)}: {game_id}")
                
                # Fetch both traditional and advanced stats
                traditional_data = self._fetch_traditional_box_score(game_id)
                advanced_data = self._fetch_advanced_box_score(game_id)
                
                if traditional_data and advanced_data:
                    # Process and merge player statistics
                    player_stats = self._merge_player_statistics(
                        game_id, traditional_data, advanced_data
                    )
                    
                    if player_stats:
                        all_player_stats.extend(player_stats)
                        successful_games += 1
                        self.success_metrics['box_scores_collected'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to collect box score for game {game_id}: {e}")
                self.failed_operations.append({
                    'type': 'box_score_collection',
                    'game_id': game_id,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                self.success_metrics['api_failures'] += 1
                continue
        
        logger.info(f"Collected box scores for {successful_games}/{len(game_ids)} games")
        return all_player_stats
    
    def upload_to_bigquery(self, table_name: str, records: List[Dict]):
        """Upload with error handling and data validation"""
        if not records:
            logger.info(f"No records to upload to {table_name}")
            return
        
        # Data validation
        validated_records = self._validate_and_clean_records(records, table_name)
        
        if not validated_records:
            logger.warning(f"No valid records after cleaning for {table_name}")
            return
        
        table_id = f"{self.client.project}.{self.dataset_id}.{table_name}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            ignore_unknown_values=True,
            max_bad_records=max(10, len(validated_records) // 100),
            create_disposition="CREATE_IF_NEEDED"
        )
        
        try:
            job = self.client.load_table_from_json(
                validated_records, table_id, job_config=job_config
            )
            job.result()
            
            logger.info(f"Successfully uploaded {len(validated_records)} records to {table_name}")
            
            if job.errors:
                logger.warning(f"Upload had {len(job.errors)} errors: {job.errors[:5]}")  # Log first 5 errors
            
        except Exception as e:
            logger.error(f"Critical error uploading to BigQuery {table_name}: {e}")
            self._save_failed_records(table_name, validated_records)
            raise
    
    def generate_collection_report(self) -> Dict:
        """Generate comprehensive collection report"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'success_metrics': self.success_metrics.copy(),
            'failed_operations': len(self.failed_operations),
            'failure_details': self.failed_operations[-10:],  # Last 10 failures
            'success_rate': {
                'api_calls': 1 - (self.success_metrics['api_failures'] / max(1, self.success_metrics['api_calls_made'])),
                'game_collection': self.success_metrics['games_collected'],
                'box_score_collection': self.success_metrics['box_scores_collected']
            }
        }