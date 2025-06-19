from nba_api.stats.endpoints import (
    boxscoretraditionalv2, playergamelog, teamgamelog,
    commonplayerinfo, commonteamroster, leaguegamefinder
)
from nba_api.stats.static import players, teams
from nba_api.live.nba.endpoints import scoreboard
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBADataClient:
    def __init__(self):
        self.rate_limit_delay = 1
    
    def get_todays_games(self) -> Dict:
        """Get today's NBA games"""
        try:
            board = scoreboard.ScoreBoard()
            time.sleep(self.rate_limit_delay)
            return board.games.get_dict()
        except Exception as e:
            logger.error(f"Error fetching today's games: {e}")
            return {}
    
    def get_recent_games(self, days_back: int = 7) -> List[Dict]:
        """Get games from the last N days"""
        games = []
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime('%m/%d/%Y')
            try:
                board = scoreboard.ScoreBoard()
                game_data = board.get_dict()
                if game_data.get('GameHeader'):
                    games.extend(game_data['GameHeader'])
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                logger.error(f"Error fetching games for {date}: {e}")
                continue
        return games
    
    def get_game_boxscore(self, game_id: str) -> Dict:
        """Get detailed boxscore for a specific game"""
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            time.sleep(self.rate_limit_delay)
            return boxscore.get_normalized_dict()
        except Exception as e:
            logger.error(f"Error fetching boxscore for game {game_id}: {e}")
            return {}
    
    def get_player_recent_games(self, player_id: int, season: str = '2024-25') -> pd.DataFrame:
        """Get recent games for a specific player"""
        try:
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            time.sleep(self.rate_limit_delay)
            return gamelog.get_data_frames()[0]
        except Exception as e:
            logger.error(f"Error fetching player games for {player_id}: {e}")
            return pd.DataFrame()
    
    def get_team_recent_games(self, team_id: int, season: str = '2024-25') -> pd.DataFrame:
        """Get recent games for a specific team"""
        try:
            gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            time.sleep(self.rate_limit_delay)
            return gamelog.get_data_frames()[0]
        except Exception as e:
            logger.error(f"Error fetching team games for {team_id}: {e}")
            return pd.DataFrame()
    
    def get_player_info(self, player_id: int) -> Dict:
        """Get player biographical information"""
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            time.sleep(self.rate_limit_delay)
            return info.get_normalized_dict()
        except Exception as e:
            logger.error(f"Error fetching player info for {player_id}: {e}")
            return {}
    
    def search_players(self, name: str) -> List[Dict]:
        """Search for players by name"""
        try:
            all_players = players.get_players()
            return [p for p in all_players if name.lower() in p['full_name'].lower()]
        except Exception as e:
            logger.error(f"Error searching for players with name '{name}': {e}")
            return []
    
    def get_all_teams(self) -> List[Dict]:
        """Get all NBA teams"""
        try:
            return teams.get_teams()
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            return []
