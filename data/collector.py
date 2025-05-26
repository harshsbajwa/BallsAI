from nba_api.live.nba.endpoints import boxscore
from nba_api.stats.endpoints import leaguegamefinder, teamgamelogs
from nba_api.stats.static import teams
from datetime import datetime, timezone
import time
import requests
from typing import List, Dict, Optional
from google.cloud import bigquery
import logging
import pandas as pd
import json
from functools import wraps
import numpy as np

from config.settings import settings

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_collection_detailed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NBADataCollector:
    def __init__(self, bigquery_client):
        self.client = bigquery_client
        self.dataset_id = settings.DATASET_ID
        self.failed_game_ids = set()
        self.retry_count = {}
        self.max_retries = 3
        self.base_delay = 5
        
        # Cache for team and player mappings
        self.team_cache = {}
        self.player_cache = {}
        self._initialize_caches()


    def _initialize_caches(self):
        """Initialize team and player caches to reduce API calls"""
        try:
            nba_teams = teams.get_teams()
            self.team_cache = {team['id']: team for team in nba_teams}
            logger.info(f"Cached {len(self.team_cache)} NBA teams")
        except Exception as e:
            logger.error(f"Failed to initialize team cache: {e}")
    

    def collect_daily_games(self) -> List[Dict]:
        """Collect today's games"""
        try:
            from nba_api.live.nba.endpoints import scoreboard
            from datetime import date
            
            today = date.today()
            board = scoreboard.ScoreBoard()
            games_data = board.get_dict()
            
            games = []
            for game in games_data.get('scoreboard', {}).get('games', []):
                game_record = {
                    'game_id': str(game['gameId']),
                    'game_date': today.strftime('%Y-%m-%d'),
                    'season': '2024-25',
                    'home_team_id': int(game['homeTeam']['teamId']),
                    'away_team_id': int(game['awayTeam']['teamId']),
                    'home_team_name': game['homeTeam'].get('teamName', ''),
                    'away_team_name': game['awayTeam'].get('teamName', ''),
                    'home_score': game['homeTeam'].get('score'),
                    'away_score': game['awayTeam'].get('score'),
                    'home_win': None if game['gameStatus'] != 3 else game['homeTeam']['score'] > game['awayTeam']['score'],
                    'game_status': game.get('gameStatusText', ''),
                    'game_status_id': game.get('gameStatus', 1),
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
                games.append(game_record)
            
            logger.info(f"Collected {len(games)} games for today")
            return games
            
        except Exception as e:
            logger.error(f"Error collecting daily games: {e}")
            return []


    def retry(self, max_retries=3, base_delay=1, backoff_factor=2):
        """Retry decorator with exponential backoff"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        result = func(*args, **kwargs)
                        if result is not None:
                            return result
                    except requests.exceptions.RequestException as e:
                        last_exception = e
                        if "429" in str(e) or "rate" in str(e).lower():
                            delay = base_delay * (backoff_factor ** attempt) + np.random.uniform(0, 2)
                            logger.warning(f"Rate limited on attempt {attempt + 1}, waiting {delay:.1f}s")
                            time.sleep(delay)
                        else:
                            logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                            time.sleep(base_delay * (attempt + 1))
                    except json.JSONDecodeError as e:
                        last_exception = e
                        logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                        time.sleep(base_delay)
                    except Exception as e:
                        last_exception = e
                        logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
                        time.sleep(base_delay)
                
                logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                if last_exception:
                    raise last_exception
                return None
            return wrapper
        return decorator


    @retry(max_retries=3, base_delay=5)
    def collect_historical_games(self, season_str: str) -> List[Dict]:
        """Historical game collection"""
        try:
            all_games = []
            logger.info(f"Collecting historical games for season: {season_str}")
            
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season_str,
                season_type_nullable='Regular Season'
            )

            games_df = gamefinder.get_data_frames()[0]
            
            if games_df.empty:
                logger.warning(f"No games found for season {season_str}")
                return []

            # Process each unique game
            unique_games = games_df.drop_duplicates(subset=['GAME_ID'])
            logger.info(f"Found {len(unique_games)} unique games")

            for idx, game_row in unique_games.iterrows():
                try:
                    # Get opponent game data
                    opponent_games = games_df[
                        (games_df['GAME_ID'] == game_row['GAME_ID']) &
                        (games_df['TEAM_ID'] != game_row['TEAM_ID'])
                    ]
                    
                    if opponent_games.empty:
                        logger.warning(f"No opponent found for game {game_row['GAME_ID']}")
                        continue
                    
                    opponent_game = opponent_games.iloc[0]

                    # Home/Away determination
                    is_home_game = 'vs.' in game_row['MATCHUP']
                    
                    home_team_id = game_row['TEAM_ID'] if is_home_game else opponent_game['TEAM_ID']
                    away_team_id = opponent_game['TEAM_ID'] if is_home_game else game_row['TEAM_ID']
                    
                    # Score handling
                    home_score = self._safe_int_convert(
                        game_row['PTS'] if is_home_game else opponent_game['PTS']
                    )
                    away_score = self._safe_int_convert(
                        opponent_game['PTS'] if is_home_game else game_row['PTS']
                    )

                    # Team name lookup
                    home_team_name = self._get_team_name(int(home_team_id))
                    away_team_name = self._get_team_name(int(away_team_id))

                    game_record = {
                        'game_id': str(game_row['GAME_ID']),
                        'game_date': pd.to_datetime(game_row['GAME_DATE']).strftime('%Y-%m-%d'),
                        'season': str(game_row['SEASON_ID']),
                        'home_team_id': int(home_team_id),
                        'away_team_id': int(away_team_id),
                        'home_team_name': home_team_name,
                        'away_team_name': away_team_name,
                        'home_score': home_score,
                        'away_score': away_score,
                        'home_win': (home_score > away_score) if (home_score is not None and away_score is not None) else None,
                        'game_status': 'Final',
                        'game_status_id': 3,
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'data_quality_score': self._calculate_game_quality_score(game_row, opponent_game)
                    }
                    all_games.append(game_record)

                except Exception as e:
                    logger.error(f"Error processing game {game_row['GAME_ID']}: {e}")
                    continue

            logger.info(f"Successfully collected {len(all_games)} games for season {season_str}")
            return all_games

        except Exception as e:
            logger.error(f"Error collecting historical games for season {season_str}: {e}")
            raise


    def _safe_int_convert(self, value) -> Optional[int]:
        """Safely convert value to integer"""
        if pd.isna(value) or value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


    def _get_team_name(self, team_id: int) -> str:
        """Get team name from cache or return Unknown"""
        team_info = self.team_cache.get(team_id)
        if team_info:
            return team_info['full_name']
        return f"Team_{team_id}"


    def _calculate_game_quality_score(self, game_row, opponent_game) -> float:
        """Calculate data quality score for a game (0-1)"""
        score = 1.0
        
        # Check for missing essential data
        if pd.isna(game_row['PTS']) or pd.isna(opponent_game['PTS']):
            score -= 0.3
        if pd.isna(game_row['GAME_DATE']):
            score -= 0.2
        if not game_row['MATCHUP'] or not opponent_game['MATCHUP']:
            score -= 0.1
            
        return max(0.0, score)


    @retry(max_retries=5, base_delay=3)
    def collect_box_scores(self, game_ids: List[str]) -> List[Dict]:
        """Box score collection"""
        player_records = []
        successful_games = 0
        
        for i, game_id in enumerate(game_ids):
            try:
                # Progressive delay based on previous failures
                delay = min(2 + (len(self.failed_game_ids) * 0.1), 10)
                time.sleep(delay)
                
                logger.debug(f"Collecting box score for game {game_id} ({i+1}/{len(game_ids)})")
                
                box = boxscore.BoxScore(game_id)
                
                # Validate box score data
                if not self._validate_box_score(box):
                    logger.warning(f"Invalid box score data for game {game_id}")
                    self.failed_game_ids.add(game_id)
                    continue
                
                # Process both teams
                for team_type in ['home', 'away']:
                    team_data = getattr(box, f'{team_type}_team').get_dict()
                    
                    for player in team_data.get('players', []):
                        if player.get('played') == '1':
                            stats = player.get('statistics', {})
                            
                            # Player record
                            player_record = self._create_player_record(
                                game_id, player, stats, team_data['teamId']
                            )
                            
                            if player_record:
                                player_records.append(player_record)

                successful_games += 1
                
                # Remove from failed set if successful
                self.failed_game_ids.discard(game_id)
                
            except Exception as e:
                error_msg = str(e)
                self.failed_game_ids.add(game_id)
                
                # Categorize errors for better handling
                if "Expecting value" in error_msg or "JSONDecodeError" in error_msg:
                    logger.warning(f"JSON parsing error for game {game_id}: {error_msg}")
                elif "429" in error_msg or "rate" in error_msg.lower():
                    logger.warning(f"Rate limited on game {game_id}, will retry")
                    time.sleep(10)  # Longer delay for rate limiting
                else:
                    logger.error(f"Unexpected error for game {game_id}: {error_msg}")
                
                continue

        logger.info(f"Successfully collected box scores for {successful_games}/{len(game_ids)} games")
        logger.info(f"Total player performances: {len(player_records)}")
        
        if self.failed_game_ids:
            logger.warning(f"Failed games: {len(self.failed_game_ids)}")
        
        return player_records


    def _validate_box_score(self, box) -> bool:
        """Validate box score data quality"""
        try:
            home_team = box.home_team.get_dict()
            away_team = box.away_team.get_dict()
            
            # Check if teams have players
            if not home_team.get('players') or not away_team.get('players'):
                return False
            
            # Check if at least some players have statistics
            home_stats_count = sum(1 for p in home_team['players'] if p.get('statistics'))
            away_stats_count = sum(1 for p in away_team['players'] if p.get('statistics'))
            
            return home_stats_count > 0 and away_stats_count > 0
            
        except Exception:
            return False


    def _create_player_record(self, game_id: str, player: dict, stats: dict, team_id: int) -> Optional[dict]:
        try:
            # Validate required fields
            if not player.get('personId') or not player.get('name'):
                return None
            
            # Statistics with conversions
            record = {
                'game_id': str(game_id),
                'player_id': int(player['personId']),
                'team_id': int(team_id),
                'player_name': str(player['name']),
                'position': player.get('position', 'Unknown'),
                'minutes': stats.get('minutes', '0:00'),
                'points': self._safe_int_convert(stats.get('points', 0)),
                'rebounds': self._safe_int_convert(stats.get('reboundsTotal', 0)),
                'assists': self._safe_int_convert(stats.get('assists', 0)),
                'steals': self._safe_int_convert(stats.get('steals', 0)),
                'blocks': self._safe_int_convert(stats.get('blocks', 0)),
                'turnovers': self._safe_int_convert(stats.get('turnovers', 0)),
                'field_goals_made': self._safe_int_convert(stats.get('fieldGoalsMade', 0)),
                'field_goals_attempted': self._safe_int_convert(stats.get('fieldGoalsAttempted', 0)),
                'three_pointers_made': self._safe_int_convert(stats.get('threePointersMade', 0)),
                'three_pointers_attempted': self._safe_int_convert(stats.get('threePointersAttempted', 0)),
                'free_throws_made': self._safe_int_convert(stats.get('freeThrowsMade', 0)),
                'free_throws_attempted': self._safe_int_convert(stats.get('freeThrowsAttempted', 0)),
                'plus_minus': self._safe_float_convert(stats.get('plusMinusPoints', 0)),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'data_quality_score': self._calculate_player_quality_score(stats)
            }
            
            return record
            
        except Exception as e:
            logger.error(f"Error creating player record for {player.get('name', 'Unknown')}: {e}")
            return None


    def _safe_float_convert(self, value) -> Optional[float]:
        if pd.isna(value) or value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


    def _calculate_player_quality_score(self, stats: dict) -> float:
        """Calculate data quality score for player stats"""
        score = 1.0
        essential_stats = ['points', 'reboundsTotal', 'assists', 'fieldGoalsMade', 'fieldGoalsAttempted']
        
        for stat in essential_stats:
            if stats.get(stat) is None:
                score -= 0.1
        
        return max(0.0, score)


    def collect_team_averages_with_history(self, season_str: str, lookback_games: int = 10) -> Dict:
        """Collect team averages with historical context"""
        try:
            team_averages = {}
            
            for team_id in self.team_cache.keys():
                try:
                    time.sleep(1)  # Rate limiting
                    
                    # Get team game logs
                    team_logs = teamgamelogs.TeamGameLogs(
                        team_id_nullable=team_id,
                        season_nullable=season_str,
                        season_type_nullable='Regular Season'
                    )
                    
                    df = team_logs.get_data_frames()[0]
                    
                    if not df.empty:
                        # Calculate rolling averages
                        recent_games = df.head(lookback_games)
                        
                        team_averages[team_id] = {
                            'avg_points': recent_games['PTS'].mean(),
                            'avg_rebounds': recent_games['REB'].mean(),
                            'avg_assists': recent_games['AST'].mean(),
                            'avg_fg_pct': recent_games['FG_PCT'].mean(),
                            'avg_3p_pct': recent_games['FG3_PCT'].mean(),
                            'games_played': len(recent_games),
                            'last_updated': datetime.now(timezone.utc).isoformat()
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not get stats for team {team_id}: {e}")
                    continue
            
            logger.info(f"Collected averages for {len(team_averages)} teams")
            return team_averages
            
        except Exception as e:
            logger.error(f"Error collecting team averages: {e}")
            return {}


    def retry_failed_games(self) -> List[Dict]:
        """Retry collecting data for previously failed games"""
        if not self.failed_game_ids:
            logger.info("No failed games to retry")
            return []
        
        logger.info(f"Retrying {len(self.failed_game_ids)} failed games")
        
        # Convert to list and clear the set
        failed_list = list(self.failed_game_ids)
        self.failed_game_ids.clear()
        
        # Retry with longer delays
        return self.collect_box_scores(failed_list)
    

    def validate_and_clean_data(self, records: List[Dict], table_type: str) -> List[Dict]:
        """Data validation and cleaning"""
        cleaned_records = []
        
        for record in records:
            try:
                # Basic validation based on table type
                if table_type == 'game_stats':
                    if not record.get('game_id') or not record.get('game_date'):
                        continue
                elif table_type == 'player_stats':
                    if not record.get('game_id') or not record.get('player_id'):
                        continue
                elif table_type == 'betting_odds':
                    if not record.get('game_id') or not record.get('book_name'):
                        continue
                
                # Clean numeric fields
                for key, value in record.items():
                    if value is None:
                        continue
                    
                    # Convert inf and -inf to None
                    if isinstance(value, (int, float)) and not np.isfinite(value):
                        record[key] = None
                
                cleaned_records.append(record)
                
            except Exception as e:
                logger.warning(f"Error cleaning record: {e}")
                continue
        
        logger.info(f"Cleaned {len(cleaned_records)}/{len(records)} records for {table_type}")
        return cleaned_records


    def upload_to_bigquery(self, table_name: str, records: List[Dict]):
        """Enhanced upload with better error handling"""
        if not records:
            logger.info(f"No records to upload to {table_name}")
            return

        # Clean and validate data
        clean_records = self.validate_and_clean_data(records, table_name)
        
        if not clean_records:
            logger.warning("No valid records to upload after cleaning")
            return

        table_id = f"{self.client.project}.{self.dataset_id}.{table_name}"

        # Avoid circular imports
        from data.schema import GAME_STATS_SCHEMA, BETTING_ODDS_SCHEMA, PLAYER_STATS_SCHEMA

        schema_map = {
            'game_stats': GAME_STATS_SCHEMA,
            'betting_odds': BETTING_ODDS_SCHEMA,
            'player_stats': PLAYER_STATS_SCHEMA
        }

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            ignore_unknown_values=True,
            schema=schema_map.get(table_name),
            max_bad_records=max(10, len(clean_records) // 100)  # Allow up to 1% bad records
        )

        try:
            job = self.client.load_table_from_json(
                clean_records, table_id, job_config=job_config
            )
            job.result()

            logger.info(f"Successfully uploaded {len(clean_records)} records to {table_name}")

            if job.errors:
                logger.warning(f"Upload had {len(job.errors)} errors: {job.errors}")

        except Exception as e:
            logger.error(f"Error uploading to BigQuery: {e}")
            # Try to save failed records locally for debugging
            try:
                import json
                with open(f'failed_upload_{table_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                    json.dump(clean_records[:10], f, indent=2, default=str)  # Save first 10 records
                logger.info("Saved sample failed records for debugging")
            except:
                pass
            raise


    def generate_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report"""
        try:
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'failed_games': len(self.failed_game_ids),
                'failed_game_ids': list(self.failed_game_ids),
                'cache_status': {
                    'teams_cached': len(self.team_cache),
                    'players_cached': len(self.player_cache)
                }
            }
            
            # Query BigQuery for additional stats
            queries = {
                'total_games': f"SELECT COUNT(*) as count FROM `{self.client.project}.{self.dataset_id}.game_stats`",
                'total_player_stats': f"SELECT COUNT(*) as count FROM `{self.client.project}.{self.dataset_id}.player_stats`",
                'games_with_null_scores': f"""
                    SELECT COUNT(*) as count 
                    FROM `{self.client.project}.{self.dataset_id}.game_stats` 
                    WHERE home_score IS NULL OR away_score IS NULL
                """,
                'player_stats_with_nulls': f"""
                    SELECT COUNT(*) as count 
                    FROM `{self.client.project}.{self.dataset_id}.player_stats` 
                    WHERE points IS NULL OR rebounds IS NULL OR assists IS NULL
                """
            }
            
            for query_name, query in queries.items():
                try:
                    result = self.client.query(query).to_dataframe()
                    report[query_name] = int(result.iloc[0]['count'])
                except Exception as e:
                    logger.warning(f"Could not execute query {query_name}: {e}")
                    report[query_name] = 'unknown'
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")
            return {'error': str(e)}
