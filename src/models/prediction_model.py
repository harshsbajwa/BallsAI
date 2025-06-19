import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging
import os
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBAGamePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super(NBAGamePredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layers
        # Winner prediction (binary classification)
        self.winner_head = nn.Sequential(
            *layers,
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        )
        
        # Score prediction (regression)
        self.score_head = nn.Sequential(
            *layers,
            nn.Linear(prev_size, 2),  # home_score, away_score
            nn.ReLU()
        )
        
    def forward(self, x):
        winner_prob = self.winner_head(x)
        scores = self.score_head(x)
        return winner_prob, scores

class NBAPlayerProjector(nn.Module):
    def __init__(self, input_size: int, output_size: int = 15):
        super(NBAPlayerProjector, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

class NBAMLPipeline:
    def __init__(self, model_path: str = "models"):
        self.model_path = model_path
        self.game_predictor = None
        self.player_projector = None
        self.feature_scaler = None
        self.team_encoder = None
        self.player_encoder = None
        self.feature_columns = None
        
    def prepare_game_features(self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for game prediction"""
        # Calculate average stats for each team across all their games
        stat_cols = [
            'team_score', 'opponent_score', 'win', 'field_goals_percentage',
            'three_pointers_percentage', 'free_throws_percentage', 'rebounds_total',
            'assists', 'turnovers', 'steals', 'blocks'
        ]
        team_avg_stats = team_stats_df.groupby('team_id')[stat_cols].mean().reset_index()

        # Merge stats for the home team
        features_df = pd.merge(
            games_df,
            team_avg_stats,
            left_on='home_team_id',
            right_on='team_id',
            suffixes=('', '_home_avg')
        )

        # Merge stats for the away team
        features_df = pd.merge(
            features_df,
            team_avg_stats,
            left_on='away_team_id',
            right_on='team_id',
            suffixes=('_home', '_away')
        )

        # Select the feature columns
        feature_cols = [col for col in features_df.columns if col.endswith(('_home', '_away'))]
        return features_df[feature_cols].fillna(0)

    def prepare_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for player projections"""
        # Rolling averages and trends
        df_sorted = df.sort_values(['person_id', 'game_date'])
        
        rolling_cols = [
            'points', 'assists', 'rebounds_total', 'steals', 'blocks',
            'field_goals_percentage', 'three_pointers_percentage', 'free_throws_percentage',
            'turnovers', 'num_minutes'
        ]
        
        # Calculate rolling statistics
        for col in rolling_cols:
            df_sorted[f'{col}_5_game_avg'] = df_sorted.groupby('person_id')[col].rolling(5, min_periods=1).mean().values
            df_sorted[f'{col}_10_game_avg'] = df_sorted.groupby('person_id')[col].rolling(10, min_periods=1).mean().values
            df_sorted[f'{col}_trend'] = df_sorted.groupby('person_id')[col].rolling(5, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            ).values
        
        # Opponent strength
        df_sorted['opp_def_rating'] = df_sorted.groupby('opponent_team_id')['points'].transform('mean')
        
        # Rest days
        df_sorted['rest_days'] = df_sorted.groupby('person_id')['game_date'].diff().dt.days.fillna(2)
        
        # Home/away performance differential
        home_avg = df_sorted[df_sorted['home'] == True].groupby('person_id')['points'].mean()
        away_avg = df_sorted[df_sorted['home'] == False].groupby('person_id')['points'].mean()
        df_sorted['home_away_diff'] = df_sorted['person_id'].map(home_avg - away_avg).fillna(0)
        
        return df_sorted
    
    def train_game_predictor(self, training_data: pd.DataFrame):
        """Train the game prediction model"""
        logger.info("Training game prediction model...")
        os.makedirs(self.model_path, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Data preparation
        game_cols = ['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score']
        games_df = training_data[game_cols].drop_duplicates().reset_index(drop=True)
        features_df = self.prepare_game_features(games_df, training_data)
        y_winner = (games_df['home_score'] > games_df['away_score']).astype(int)
        y_scores = games_df[['home_score', 'away_score']].values
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(features_df.values)
        X_train, X_test, y_winner_train, y_winner_test, y_scores_train, y_scores_test = train_test_split(
            X_scaled, y_winner, y_scores, test_size=0.2, random_state=42
        )

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_winner_train_tensor = torch.FloatTensor(y_winner_train.values).unsqueeze(1)
        y_scores_train_tensor = torch.FloatTensor(y_scores_train)
        
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_winner_test_tensor = torch.FloatTensor(y_winner_test.values).unsqueeze(1).to(device)
        y_scores_test_tensor = torch.FloatTensor(y_scores_test).to(device)

        # Initialize model
        self.game_predictor = NBAGamePredictor(input_size=X_train.shape[1])
        self.game_predictor.to(device)

        # Training setup
        criterion_winner = nn.BCELoss()
        criterion_scores = nn.MSELoss()
        optimizer = torch.optim.Adam(self.game_predictor.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        # Data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_winner_train_tensor, y_scores_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(200):
            self.game_predictor.train()
            
            # Iterate over batches
            for batch_X, batch_y_winner, batch_y_scores in train_loader:
                batch_X = batch_X.to(device)
                batch_y_winner = batch_y_winner.to(device)
                batch_y_scores = batch_y_scores.to(device)

                # Forward pass
                winner_pred, scores_pred = self.game_predictor(batch_X)

                # Calculate losses
                winner_loss = criterion_winner(winner_pred, batch_y_winner)
                scores_loss = criterion_scores(scores_pred, batch_y_scores)
                total_loss = winner_loss + 0.1 * scores_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Validation
            if epoch % 10 == 0:
                self.game_predictor.eval()
                with torch.no_grad():
                    winner_pred_val, scores_pred_val = self.game_predictor(X_test_tensor)
                    
                    val_winner_loss = criterion_winner(winner_pred_val, y_winner_test_tensor)
                    val_scores_loss = criterion_scores(scores_pred_val, y_scores_test_tensor)
                    val_total_loss = val_winner_loss + 0.1 * val_scores_loss
                    
                    logger.info(f"Epoch {epoch}: Train Loss: {total_loss:.4f}, Val Loss: {val_total_loss:.4f}")
                    
                    if val_total_loss < best_loss:
                        best_loss = val_total_loss
                        patience_counter = 0
                        torch.save(self.game_predictor.state_dict(), os.path.join(self.model_path, "game_predictor_best.pth"))
                    else:
                        patience_counter += 1
                        if patience_counter >= 20:
                            logger.info("Early stopping triggered")
                            break
                    
                    scheduler.step(val_total_loss)

        # Save final model and preprocessors
        torch.save(self.game_predictor.state_dict(), os.path.join(self.model_path, "game_predictor_final.pth"))
        joblib.dump(self.feature_scaler, os.path.join(self.model_path, "feature_scaler.pkl"))
        logger.info("Game predictor training completed")

    def train_player_projector(self, training_data: pd.DataFrame):
        logger.info("Training player projection model...")
        os.makedirs(self.model_path, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Ensure the output directory exists
        os.makedirs(self.model_path, exist_ok=True)
        
        # Prepare features and targets
        features_df = self.prepare_player_features(training_data)
        
        # Feature columns
        feature_cols = [col for col in features_df.columns if any(suffix in col for suffix in 
                       ['_5_game_avg', '_10_game_avg', '_trend', 'opp_def_rating', 'rest_days', 'home_away_diff'])]
        
        # Target columns
        target_cols = [
            'points', 'assists', 'rebounds_total', 'steals', 'blocks',
            'field_goals_made', 'field_goals_attempted', 'three_pointers_made', 'three_pointers_attempted',
            'free_throws_made', 'free_throws_attempted', 'turnovers', 'fouls_personal', 'num_minutes',
            'plus_minus_points'
        ]
        
        # Prepare data
        X = features_df[feature_cols].fillna(0).values
        y = features_df[target_cols].fillna(0).values
        
        # Scale features
        player_feature_scaler = StandardScaler()
        X_scaled = player_feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        # Initialize model
        self.player_projector = NBAPlayerProjector(
            input_size=X_train.shape[1], 
            output_size=len(target_cols)
        )
        self.player_projector.to(device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.player_projector.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(300):
            self.player_projector.train()
            
            # Iterate over batches
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                predictions = self.player_projector(batch_X)
                loss = criterion(predictions, batch_y)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            if epoch % 20 == 0:
                self.player_projector.eval()
                with torch.no_grad():
                    val_predictions = self.player_projector(X_test_tensor)
                    val_loss = criterion(val_predictions, y_test_tensor)
                    
                    logger.info(f"Epoch {epoch}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        torch.save(self.player_projector.state_dict(), os.path.join(self.model_path, "player_projector_best.pth"))
                    else:
                        patience_counter += 1
                        if patience_counter >= 15:
                            logger.info("Early stopping triggered")
                            break
                    
                    scheduler.step(val_loss)
        
        # Save final model and preprocessors
        torch.save(self.player_projector.state_dict(), os.path.join(self.model_path, "player_projector_final.pth"))
        joblib.dump(player_feature_scaler, os.path.join(self.model_path, "player_feature_scaler.pkl"))
        joblib.dump(feature_cols, os.path.join(self.model_path, "player_feature_cols.pkl"))
        
        logger.info("Player projector training completed")
    
    def load_models(self):
        """Load trained models"""
        try:
            # Deployment on CPU
            device = torch.device('cpu')
            logger.info(f"Loading models to device: {device}")

            # Load Game Predictor
            self.game_predictor = NBAGamePredictor(input_size=24)
            game_predictor_path = os.path.join(self.model_path, "game_predictor_best.pth")
            state_dict = torch.load(game_predictor_path, map_location=device)
            self.game_predictor.load_state_dict(state_dict)
            self.game_predictor.eval()
            self.feature_scaler = joblib.load(os.path.join(self.model_path, "feature_scaler.pkl"))

            # Load Player Projector
            self.feature_columns = joblib.load(os.path.join(self.model_path, "player_feature_cols.pkl"))
            player_input_size = len(self.feature_columns)
            self.player_projector = NBAPlayerProjector(input_size=player_input_size, output_size=15)
            player_projector_path = os.path.join(self.model_path, "player_projector_best.pth")
            state_dict = torch.load(player_projector_path, map_location=device)
            self.player_projector.load_state_dict(state_dict)
            self.player_projector.eval()
            self.player_feature_scaler = joblib.load(os.path.join(self.model_path, "player_feature_scaler.pkl"))
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict_game_outcome(self, home_team_features: Dict, away_team_features: Dict) -> Dict:
        """Predict game outcome"""
        if self.game_predictor is None:
            raise ValueError("Game predictor not loaded")
        
        # Prepare features
        features = np.concatenate([
            list(home_team_features.values()),
            list(away_team_features.values())
        ])
        
        # Scale features
        features_scaled = self.feature_scaler.transform([features])
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Predict
        with torch.no_grad():
            winner_prob, scores = self.game_predictor(features_tensor)
            
        return {
            'home_win_probability': float(winner_prob[0][0]),
            'away_win_probability': 1 - float(winner_prob[0][0]),
            'predicted_home_score': float(scores[0][0]),
            'predicted_away_score': float(scores[0][1]),
        }
    
    def predict_player_stats(self, player_features: Dict) -> Dict:
        """Predict player statistics"""
        if self.player_projector is None:
            raise ValueError("Player projector not loaded")
        
        # Prepare features
        features = np.array(list(player_features.values())).reshape(1, -1)
        features_scaled = self.player_feature_scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled)
        
        # Predict
        with torch.no_grad():
            predictions = self.player_projector(features_tensor)
        
        # Map predictions to stat names
        stat_names = [
            'points', 'assists', 'rebounds_total', 'steals', 'blocks',
            'field_goals_made', 'field_goals_attempted', 'three_pointers_made', 'three_pointers_attempted',
            'free_throws_made', 'free_throws_attempted', 'turnovers', 'fouls_personal', 'num_minutes',
            'plus_minus_points'
        ]
        
        return dict(zip(stat_names, predictions[0].tolist()))