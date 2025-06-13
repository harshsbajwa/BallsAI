import pandas as pd
from sqlalchemy.orm import Session
from src.models.database import Game, PlayerStatistic, TeamStatistic, SessionLocal
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self):
        self.db = SessionLocal()
    
    def process_daily_data(self, raw_data: dict, execution_date: str):
        """Process and load daily NBA data"""
        try:
            self._process_games(raw_data.get('recent_games', []))
            
            self._process_boxscores(raw_data.get('boxscores', []))
            
            logger.info(f"Successfully processed daily data for {execution_date}")
        except Exception as e:
            logger.error(f"Error processing daily data: {e}")
            self.db.rollback()
            raise
        finally:
            self.db.close()
    
    def _process_games(self, games_data: list):
        """Process and upsert games data"""
        for game in games_data:
            existing_game = self.db.query(Game).filter(
                Game.game_id == game.get('GAME_ID')
            ).first()
            
            if not existing_game:
                new_game = Game(
                    game_id=game.get('GAME_ID'),
                    game_date=datetime.strptime(game.get('GAME_DATE_EST'), '%Y-%m-%dT%H:%M:%S'),
                    home_team_id=game.get('HOME_TEAM_ID'),
                    away_team_id=game.get('VISITOR_TEAM_ID'),
                    home_score=game.get('PTS_HOME'),
                    away_score=game.get('PTS_AWAY'),
                    game_type='Regular Season',
                )
                self.db.add(new_game)
        
        self.db.commit()
    
    def _process_boxscores(self, boxscores_data: list):
        """Process boxscore data for player and team statistics"""
        for boxscore_entry in boxscores_data:
            game_id = boxscore_entry.get('game_id')
            boxscore = boxscore_entry.get('boxscore', {})
            
            if 'PlayerStats' in boxscore:
                self._process_player_stats(game_id, boxscore['PlayerStats'])
            
            if 'TeamStats' in boxscore:
                self._process_team_stats(game_id, boxscore['TeamStats'])
    
    def _process_player_stats(self, game_id: str, player_stats: list):
        """Process player statistics from boxscore"""
        for stat in player_stats:
            existing_stat = self.db.query(PlayerStatistic).filter(
                PlayerStatistic.game_id == game_id,
                PlayerStatistic.person_id == stat.get('PLAYER_ID')
            ).first()
            
            if not existing_stat:
                new_stat = PlayerStatistic(
                    person_id=stat.get('PLAYER_ID'),
                    game_id=game_id,
                    game_date=datetime.now(),
                    team_id=stat.get('TEAM_ID'),
                    points=stat.get('PTS', 0),
                    assists=stat.get('AST', 0),
                    rebounds_total=stat.get('REB', 0),
                    steals=stat.get('STL', 0),
                    blocks=stat.get('BLK', 0),
                    field_goals_made=stat.get('FGM', 0),
                    field_goals_attempted=stat.get('FGA', 0),
                    three_pointers_made=stat.get('FG3M', 0),
                    three_pointers_attempted=stat.get('FG3A', 0),
                    free_throws_made=stat.get('FTM', 0),
                    free_throws_attempted=stat.get('FTA', 0),
                    fouls_personal=stat.get('PF', 0),
                    turnovers=stat.get('TO', 0),
                    plus_minus_points=stat.get('PLUS_MINUS', 0),
                )
                self.db.add(new_stat)
        
        self.db.commit()
    
    def _process_team_stats(self, game_id: str, team_stats: list):
        """Process team statistics from boxscore"""
        for stat in team_stats:
            existing_stat = self.db.query(TeamStatistic).filter(
                TeamStatistic.game_id == game_id,
                TeamStatistic.team_id == stat.get('TEAM_ID')
            ).first()
            
            if not existing_stat:
                new_stat = TeamStatistic(
                    game_id=game_id,
                    team_id=stat.get('TEAM_ID'),
                    game_date=datetime.now(),
                    team_score=stat.get('PTS', 0),
                    assists=stat.get('AST', 0),
                    rebounds_total=stat.get('REB', 0),
                    steals=stat.get('STL', 0),
                    blocks=stat.get('BLK', 0),
                    field_goals_made=stat.get('FGM', 0),
                    field_goals_attempted=stat.get('FGA', 0),
                    three_pointers_made=stat.get('FG3M', 0),
                    three_pointers_attempted=stat.get('FG3A', 0),
                    free_throws_made=stat.get('FTM', 0),
                    free_throws_attempted=stat.get('FTA', 0),
                    fouls_personal=stat.get('PF', 0),
                    turnovers=stat.get('TO', 0),
                )
                self.db.add(new_stat)
        
        self.db.commit()
