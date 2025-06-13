import pandas as pd
from sqlalchemy.orm import Session
from src.models.database import Player, Team, Game, PlayerStatistic, TeamStatistic, SessionLocal
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path: str = "/data"):
        self.data_path = data_path
        self.team_lookup = {}
        
    def load_initial_data(self):
        """Load all initial historical data into PostgreSQL"""
        db = SessionLocal()
        try:
            self._load_players(db)
            self._load_teams(db)
            self._build_team_lookup(db)
            self._load_games(db)
            self._load_player_statistics(db)
            self._load_team_statistics(db)
            logger.info("Initial data loading completed successfully")
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def _load_players(self, db: Session):
        """Load players data"""
        logger.info("Loading players data...")
        df = pd.read_csv(f"{self.data_path}/Players.csv")
        
        players = []
        for _, row in df.iterrows():
            player = Player(
                person_id=row['personId'],
                first_name=row['firstName'],
                last_name=row['lastName'],
                birthdate=pd.to_datetime(row['birthdate']) if pd.notna(row['birthdate']) else None,
                last_attended=row['lastAttended'] if pd.notna(row['lastAttended']) else None,
                country=row['country'] if pd.notna(row['country']) else None,
                height=row['height'] if pd.notna(row['height']) else None,
                body_weight=row['bodyWeight'] if pd.notna(row['bodyWeight']) else None,
                guard=bool(row['guard']) if pd.notna(row['guard']) else False,
                forward=bool(row['forward']) if pd.notna(row['forward']) else False,
                center=bool(row['center']) if pd.notna(row['center']) else False,
                draft_year=int(row['draftYear']) if pd.notna(row['draftYear']) else None,
                draft_round=int(row['draftRound']) if pd.notna(row['draftRound']) else None,
                draft_number=int(row['draftNumber']) if pd.notna(row['draftNumber']) else None,
            )
            players.append(player)
        
        db.bulk_save_objects(players)
        db.commit()
        logger.info(f"Loaded {len(players)} players")
    
    def _load_teams(self, db: Session):
        """Load teams data"""
        logger.info("Loading teams data...")
        df = pd.read_csv(f"{self.data_path}/TeamHistories.csv")
        df_sorted = df.sort_values(by=['teamId', 'seasonActiveTill'], ascending=[True, False])
        df_unique_teams = df_sorted.drop_duplicates(subset='teamId', keep='first')

        teams = []
        for _, row in df_unique_teams.iterrows():
            team = Team(
                team_id=row['teamId'],
                team_city=row['teamCity'],
                team_name=row['teamName'],
                team_abbrev=row['teamAbbrev'].strip() if pd.notna(row['teamAbbrev']) else None,
                season_founded=int(row['seasonFounded']) if pd.notna(row['seasonFounded']) else None,
                season_active_till=int(row['seasonActiveTill']) if pd.notna(row['seasonActiveTill']) else None,
                league=row['league'] if pd.notna(row['league']) else None,
            )
            teams.append(team)

        db.bulk_save_objects(teams)
        db.commit()
        logger.info(f"Loaded {len(teams)} unique teams")

    def _build_team_lookup(self, db: Session):
        """Build an in-memory lookup map for team names to IDs"""
        logger.info("Building team lookup map...")
        teams = db.query(Team).all()
        self.team_lookup = {(team.team_city, team.team_name): team.team_id for team in teams}
        logger.info(f"Team lookup map built with {len(self.team_lookup)} entries.")

    def _load_games(self, db: Session):
        """Load games data"""
        logger.info("Loading games data...")
        df = pd.read_csv(f"{self.data_path}/Games.csv")
        
        games = []
        for _, row in df.iterrows():
            game = Game(
                game_id=row['gameId'],
                game_date=pd.to_datetime(row['gameDate']),
                home_team_id=row['hometeamId'],
                away_team_id=row['awayteamId'],
                home_score=int(row['homeScore']) if pd.notna(row['homeScore']) else None,
                away_score=int(row['awayScore']) if pd.notna(row['awayScore']) else None,
                winner_team_id=int(row['winner']) if pd.notna(row['winner']) else None,
                game_type=row['gameType'] if pd.notna(row['gameType']) else None,
                attendance=int(row['attendance']) if pd.notna(row['attendance']) else None,
                arena_id=int(row['arenaId']) if pd.notna(row['arenaId']) else None,
                game_label=row['gameLabel'] if pd.notna(row['gameLabel']) else None,
                game_sub_label=row['gameSubLabel'] if pd.notna(row['gameSubLabel']) else None,
                series_game_number=int(row['seriesGameNumber']) if pd.notna(row['seriesGameNumber']) else None,
            )
            games.append(game)
        
        db.bulk_save_objects(games)
        db.commit()
        logger.info(f"Loaded {len(games)} games")
    
    def _load_player_statistics(self, db: Session):
        """Load player statistics data"""
        logger.info("Loading player statistics data...")
        df = pd.read_csv(f"{self.data_path}/PlayerStatistics.csv", dtype=str, low_memory=False)

        chunk_size = 10000
        total_loaded = 0

        for chunk_start in range(0, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk_df = df.iloc[chunk_start:chunk_end]

            player_stats = []
            for _, row in chunk_df.iterrows():
                player_team_key = (row['playerteamCity'], row['playerteamName'])
                opponent_team_key = (row['opponentteamCity'], row['opponentteamName'])

                player_team_id = self.team_lookup.get(player_team_key)
                opponent_team_id = self.team_lookup.get(opponent_team_key)

                if not player_team_id or not opponent_team_id:
                    continue

                stat = PlayerStatistic(
                    person_id=int(row['personId']),
                    game_id=int(row['gameId']),
                    game_date=pd.to_datetime(row['gameDate']),
                    team_id=player_team_id,
                    opponent_team_id=opponent_team_id,
                    win=bool(int(float(row['win']))) if pd.notna(row['win']) else False,
                    home=bool(int(float(row['home']))) if pd.notna(row['home']) else False,
                    num_minutes=float(row['numMinutes']) if pd.notna(row['numMinutes']) else 0.0,
                    points=int(float(row['points'])) if pd.notna(row['points']) else 0,
                    assists=int(float(row['assists'])) if pd.notna(row['assists']) else 0,
                    blocks=int(float(row['blocks'])) if pd.notna(row['blocks']) else 0,
                    steals=int(float(row['steals'])) if pd.notna(row['steals']) else 0,
                    field_goals_attempted=int(float(row['fieldGoalsAttempted'])) if pd.notna(row['fieldGoalsAttempted']) else 0,
                    field_goals_made=int(float(row['fieldGoalsMade'])) if pd.notna(row['fieldGoalsMade']) else 0,
                    field_goals_percentage=float(row['fieldGoalsPercentage']) if pd.notna(row['fieldGoalsPercentage']) else 0.0,
                    three_pointers_attempted=int(float(row['threePointersAttempted'])) if pd.notna(row['threePointersAttempted']) else 0,
                    three_pointers_made=int(float(row['threePointersMade'])) if pd.notna(row['threePointersMade']) else 0,
                    three_pointers_percentage=float(row['threePointersPercentage']) if pd.notna(row['threePointersPercentage']) else 0.0,
                    free_throws_attempted=int(float(row['freeThrowsAttempted'])) if pd.notna(row['freeThrowsAttempted']) else 0,
                    free_throws_made=int(float(row['freeThrowsMade'])) if pd.notna(row['freeThrowsMade']) else 0,
                    free_throws_percentage=float(row['freeThrowsPercentage']) if pd.notna(row['freeThrowsPercentage']) else 0.0,
                    rebounds_defensive=int(float(row['reboundsDefensive'])) if pd.notna(row['reboundsDefensive']) else 0,
                    rebounds_offensive=int(float(row['reboundsOffensive'])) if pd.notna(row['reboundsOffensive']) else 0,
                    rebounds_total=int(float(row['reboundsTotal'])) if pd.notna(row['reboundsTotal']) else 0,
                    fouls_personal=int(float(row['foulsPersonal'])) if pd.notna(row['foulsPersonal']) else 0,
                    turnovers=int(float(row['turnovers'])) if pd.notna(row['turnovers']) else 0,
                    plus_minus_points=int(float(row['plusMinusPoints'])) if pd.notna(row['plusMinusPoints']) else 0,
                )
                player_stats.append(stat)

            db.bulk_save_objects(player_stats)
            db.commit()
            total_loaded += len(player_stats)
            logger.info(f"Loaded {total_loaded} player statistics records...")

        logger.info(f"Completed loading {total_loaded} player statistics records")

    def _load_team_statistics(self, db: Session):
        """Load team statistics data"""
        logger.info("Loading team statistics data...")
        df = pd.read_csv(f"{self.data_path}/TeamStatistics.csv")
        
        chunk_size = 5000
        total_loaded = 0
        
        for chunk_start in range(0, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk_df = df.iloc[chunk_start:chunk_end]
            
            team_stats = []
            for _, row in chunk_df.iterrows():
                stat = TeamStatistic(
                    game_id=row['gameId'],
                    team_id=row['teamId'],
                    opponent_team_id=row['opponentTeamId'],
                    game_date=pd.to_datetime(row['gameDate']),
                    home=bool(row['home']) if pd.notna(row['home']) else False,
                    win=bool(row['win']) if pd.notna(row['win']) else False,
                    team_score=int(row['teamScore']) if pd.notna(row['teamScore']) else 0,
                    opponent_score=int(row['opponentScore']) if pd.notna(row['opponentScore']) else 0,
                    assists=int(row['assists']) if pd.notna(row['assists']) else 0,
                    blocks=int(row['blocks']) if pd.notna(row['blocks']) else 0,
                    steals=int(row['steals']) if pd.notna(row['steals']) else 0,
                    field_goals_attempted=int(row['fieldGoalsAttempted']) if pd.notna(row['fieldGoalsAttempted']) else 0,
                    field_goals_made=int(row['fieldGoalsMade']) if pd.notna(row['fieldGoalsMade']) else 0,
                    field_goals_percentage=float(row['fieldGoalsPercentage']) if pd.notna(row['fieldGoalsPercentage']) else 0.0,
                    three_pointers_attempted=int(row['threePointersAttempted']) if pd.notna(row['threePointersAttempted']) else 0,
                    three_pointers_made=int(row['threePointersMade']) if pd.notna(row['threePointersMade']) else 0,
                    three_pointers_percentage=float(row['threePointersPercentage']) if pd.notna(row['threePointersPercentage']) else 0.0,
                    free_throws_attempted=int(row['freeThrowsAttempted']) if pd.notna(row['freeThrowsAttempted']) else 0,
                    free_throws_made=int(row['freeThrowsMade']) if pd.notna(row['freeThrowsMade']) else 0,
                    free_throws_percentage=float(row['freeThrowsPercentage']) if pd.notna(row['freeThrowsPercentage']) else 0.0,
                    rebounds_defensive=int(row['reboundsDefensive']) if pd.notna(row['reboundsDefensive']) else 0,
                    rebounds_offensive=int(row['reboundsOffensive']) if pd.notna(row['reboundsOffensive']) else 0,
                    rebounds_total=int(row['reboundsTotal']) if pd.notna(row['reboundsTotal']) else 0,
                    fouls_personal=int(row['foulsPersonal']) if pd.notna(row['foulsPersonal']) else 0,
                    turnovers=int(row['turnovers']) if pd.notna(row['turnovers']) else 0,
                    plus_minus_points=int(row['plusMinusPoints']) if pd.notna(row['plusMinusPoints']) else 0,
                )
                team_stats.append(stat)
            
            db.bulk_save_objects(team_stats)
            db.commit()
            total_loaded += len(team_stats)
            logger.info(f"Loaded {total_loaded} team statistics records...")
        
        logger.info(f"Completed loading {total_loaded} team statistics records")

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_initial_data()
