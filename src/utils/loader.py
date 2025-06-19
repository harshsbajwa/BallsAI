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
            self._build_team_lookup()
            self._load_games(db)
            self._load_player_statistics_optimized(db)
            self._load_team_statistics_optimized(db)
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

        current_year = datetime.now().year
        df_nba_teams = df[
            (df['league'] == 'NBA') &
            (df['seasonActiveTill'] >= current_year - 1)
        ]
        df_sorted = df_nba_teams.sort_values(by=['teamId', 'seasonActiveTill'], ascending=[True, False])
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

    def _build_team_lookup(self):
        """Build an in-memory lookup map"""
        logger.info("Building comprehensive team lookup map from source file...")
        df_history = pd.read_csv(f"{self.data_path}/TeamHistories.csv")
        
        self.team_lookup = {}
        for _, row in df_history.iterrows():
            self.team_lookup[(row['teamCity'], row['teamName'])] = row['teamId']
            
        logger.info(f"Team lookup map built with {len(self.team_lookup)} entries.")

    def _load_games(self, db: Session):
        """Load games data"""
        logger.info("Loading games data...")
        df = pd.read_csv(f"{self.data_path}/Games.csv", low_memory=False)
        
        games = []
        for _, row in df.iterrows():
            game = Game(
                game_id=row['gameId'],
                game_date=pd.to_datetime(row['gameDate']),
                home_team_id=row['hometeamId'],
                away_team_id=row['awayteamId'],
                home_score=int(row['homeScore']) if pd.notna(row['homeScore']) else None,
                away_score=int(row['awayScore']) if pd.notna(row['awayScore']) else None,
                winning_team_id=int(row['winner']) if pd.notna(row['winner']) else None,
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
    
    def _load_player_statistics_optimized(self, db: Session):
        """Load player statistics data"""
        logger.info("Loading player statistics data...")
        
        chunk_size = 100000
        total_loaded = 0
        
        team_map = pd.Series(self.team_lookup.values(), index=self.team_lookup.keys())

        for chunk_df in pd.read_csv(f"{self.data_path}/PlayerStatistics.csv", dtype=str, low_memory=False, chunksize=chunk_size):
            
            player_team_key = list(zip(chunk_df['playerteamCity'], chunk_df['playerteamName']))
            opponent_team_key = list(zip(chunk_df['opponentteamCity'], chunk_df['opponentteamName']))
            chunk_df['team_id'] = team_map.loc[player_team_key].values
            chunk_df['opponent_team_id'] = team_map.loc[opponent_team_key].values
            
            chunk_df.dropna(subset=['team_id', 'opponent_team_id'], inplace=True)

            rename_map = {
                'personId': 'person_id', 'gameId': 'game_id', 'gameDate': 'game_date',
                'win': 'win', 'home': 'home', 'numMinutes': 'num_minutes', 'points': 'points',
                'assists': 'assists', 'blocks': 'blocks', 'steals': 'steals',
                'fieldGoalsAttempted': 'field_goals_attempted', 'fieldGoalsMade': 'field_goals_made',
                'fieldGoalsPercentage': 'field_goals_percentage', 'threePointersAttempted': 'three_pointers_attempted',
                'threePointersMade': 'three_pointers_made', 'threePointersPercentage': 'three_pointers_percentage',
                'freeThrowsAttempted': 'free_throws_attempted', 'freeThrowsMade': 'free_throws_made',
                'freeThrowsPercentage': 'free_throws_percentage', 'reboundsDefensive': 'rebounds_defensive',
                'reboundsOffensive': 'rebounds_offensive', 'reboundsTotal': 'rebounds_total',
                'foulsPersonal': 'fouls_personal', 'turnovers': 'turnovers', 'plusMinusPoints': 'plus_minus_points'
            }
            chunk_df.rename(columns=rename_map, inplace=True)

            numeric_cols = list(rename_map.values())[5:]
            for col in numeric_cols:
                chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(0)

            chunk_df['game_date'] = pd.to_datetime(chunk_df['game_date'])
            chunk_df['win'] = chunk_df['win'].astype(float).astype(bool)
            chunk_df['home'] = chunk_df['home'].astype(float).astype(bool)
            
            records = chunk_df[list(rename_map.values()) + ['team_id', 'opponent_team_id']].to_dict(orient='records')
            
            if records:
                db.bulk_insert_mappings(PlayerStatistic, records)
                db.commit()
                total_loaded += len(records)
                logger.info(f"Loaded {total_loaded} player statistics records...")

        logger.info(f"Completed loading {total_loaded} player statistics records")

    def _load_team_statistics_optimized(self, db: Session):
        """Load team statistics data"""
        logger.info("Loading team statistics data...")
        chunk_size = 50000
        total_loaded = 0

        for chunk_df in pd.read_csv(f"{self.data_path}/TeamStatistics.csv", chunksize=chunk_size):
            rename_map = {
                'gameId': 'game_id', 'teamId': 'team_id', 'opponentTeamId': 'opponent_team_id',
                'gameDate': 'game_date', 'home': 'home', 'win': 'win', 'teamScore': 'team_score',
                'opponentScore': 'opponent_score', 'assists': 'assists', 'blocks': 'blocks',
                'steals': 'steals', 'fieldGoalsAttempted': 'field_goals_attempted',
                'fieldGoalsMade': 'field_goals_made', 'fieldGoalsPercentage': 'field_goals_percentage',
                'threePointersAttempted': 'three_pointers_attempted', 'threePointersMade': 'three_pointers_made',
                'threePointersPercentage': 'three_pointers_percentage', 'freeThrowsAttempted': 'free_throws_attempted',
                'freeThrowsMade': 'free_throws_made', 'freeThrowsPercentage': 'free_throws_percentage',
                'reboundsDefensive': 'rebounds_defensive', 'reboundsOffensive': 'rebounds_offensive',
                'reboundsTotal': 'rebounds_total', 'foulsPersonal': 'fouls_personal',
                'turnovers': 'turnovers', 'plusMinusPoints': 'plus_minus_points'
            }
            chunk_df.rename(columns=rename_map, inplace=True)
            
            numeric_cols = list(rename_map.values())[6:]
            for col in numeric_cols:
                chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(0)

            chunk_df['game_date'] = pd.to_datetime(chunk_df['game_date'])
            chunk_df['win'] = chunk_df['win'].astype(bool)
            chunk_df['home'] = chunk_df['home'].astype(bool)

            records = chunk_df[list(rename_map.values())].to_dict(orient='records')
            
            if records:
                db.bulk_insert_mappings(TeamStatistic, records)
                db.commit()
                total_loaded += len(records)
                logger.info(f"Loaded {total_loaded} team statistics records...")
        
        logger.info(f"Completed loading {total_loaded} team statistics records")

if __name__ == "__main__":
    loader = DataLoader()
    loader.load_initial_data()