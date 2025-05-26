from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import os
from datetime import datetime, timedelta
import joblib
from config.settings import settings


logger = logging.getLogger(__name__)

class NBADataset(Dataset):
    def __init__(self, features, targets, scaler=None, fit_scaler=True):
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
            
        if fit_scaler:
            self.features = torch.FloatTensor(self.scaler.fit_transform(features))
        else:
            self.features = torch.FloatTensor(self.scaler.transform(features))
            
        self.targets = torch.FloatTensor(targets)
    

    def __len__(self):
        return len(self.features)
    

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class WinLossModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=0.3):
        super(WinLossModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate if i < len(hidden_sizes) - 1 else dropout_rate/2)
            ])
            prev_size = hidden_size
        
        layers.extend([
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.network(x)


class SpreadModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=0.3):
        super(SpreadModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate if i < len(hidden_sizes) - 1 else dropout_rate/2)
            ])
            prev_size = hidden_size
        
        layers.extend([
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        ])
        
        self.network = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.network(x)


class NBAModelTrainer:
    def __init__(self, bigquery_client):
        self.client = bigquery_client
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        

    def load_training_data(self, min_date=None):
        """Load training data from BigQuery"""
        if min_date is None:
            min_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT *
        FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.feature_set`
        WHERE game_date >= '{min_date}'
        AND home_win IS NOT NULL
        AND home_score IS NOT NULL
        AND away_score IS NOT NULL
        ORDER BY game_date
        """
        
        logger.info("Loading training data from BigQuery...")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} records")
        
        return df
    

    def prepare_features(self, df):
        """Prepare feature columns for training"""
        # Define feature columns
        feature_prefixes = [
            'home_team_points', 'home_team_rebounds', 'home_team_assists',
            'home_team_fg_pct', 'home_team_3p_pct', 'home_effective_fg_pct',
            'home_true_shooting_pct', 'home_assist_turnover_ratio',
            'away_team_points', 'away_team_rebounds', 'away_team_assists', 
            'away_team_fg_pct', 'away_team_3p_pct', 'away_effective_fg_pct',
            'away_true_shooting_pct', 'away_assist_turnover_ratio',
            'home_rest_days', 'away_rest_days', 'rest_advantage',
            'home_back_to_back', 'away_back_to_back',
            'home_h2h_win_pct', 'home_h2h_margin_avg',
            'day_of_week', 'month', 'is_weekend', 'season_progress'
        ]
        
        # Find available feature columns
        available_features = []
        for prefix in feature_prefixes:
            matching_cols = [col for col in df.columns if col.startswith(prefix)]
            available_features.extend(matching_cols)
        
        # Remove duplicates and ensure columns exist
        feature_columns = list(set(available_features))
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        logger.info(f"Using {len(feature_columns)} feature columns")
        return feature_columns
    

    def train_model(self, model, train_loader, val_loader, criterion, optimizer, 
                   num_epochs=100, patience=15):
        """Train a model with early stopping"""
        model = model.to(self.device)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            for features, targets in train_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    if len(targets.shape) == 1:
                        targets = targets.unsqueeze(1)
                    
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                logger.info(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, train_losses, val_losses
    

    def evaluate_model(self, model, test_loader, is_classification=True):
        """Evaluate model performance"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                outputs = model(features)
                
                if is_classification:
                    probabilities = outputs.cpu().numpy().flatten()
                    predictions = (probabilities > 0.5).astype(int)
                    all_probabilities.extend(probabilities)
                    all_predictions.extend(predictions)
                else:
                    predictions = outputs.cpu().numpy().flatten()
                    all_predictions.extend(predictions)
                
                all_targets.extend(targets.numpy())
        
        if is_classification:
            metrics = {
                'accuracy': accuracy_score(all_targets, all_predictions),
                'precision': precision_score(all_targets, all_predictions),
                'recall': recall_score(all_targets, all_predictions),
                'f1': f1_score(all_targets, all_predictions),
                'auc': roc_auc_score(all_targets, all_probabilities)
            }
        else:
            mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
            rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets))**2))
            metrics = {'mae': mae, 'rmse': rmse}
        
        return metrics, all_predictions, all_targets
    

    def save_model_artifacts(self, model_data):
        """Save models and scalers with proper versioning"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save main model file
            model_path = self.model_dir / f"nba_models_{timestamp}.pth"
            torch.save(model_data, model_path)
            
            # Create symlink to latest
            latest_path = self.model_dir / "nba_models_latest.pth"
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(model_path.name)
            
            # Save scalers separately for easier access
            scaler_dir = self.model_dir / "scalers"
            scaler_dir.mkdir(exist_ok=True)
            
            joblib.dump(model_data['win_loss_scaler'], scaler_dir / f"win_loss_scaler_{timestamp}.pkl")
            joblib.dump(model_data['spread_scaler'], scaler_dir / f"spread_scaler_{timestamp}.pkl")
            
            logger.info(f"Model artifacts saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise
    

    def train_all_models(self):
        """Train both win/loss and spread models"""
        try:
            # Load and prepare data
            df = self.load_training_data()
            if df.empty:
                raise ValueError("No training data available")
            
            feature_columns = self.prepare_features(df)
            if not feature_columns:
                raise ValueError("No valid features found")
            
            logger.info(f"Training with {len(feature_columns)} features on {len(df)} samples")
            
            # Create time-based split
            df = df.sort_values('game_date')
            split_date = df['game_date'].quantile(0.8)
            train_mask = df['game_date'] <= split_date
            test_mask = df['game_date'] > split_date
            
            X = df[feature_columns].fillna(0).values
            y_win_loss = df['home_win'].astype(int).values
            y_spread = (df['home_score'] - df['away_score']).values
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_win_train, y_win_test = y_win_loss[train_mask], y_win_loss[test_mask]
            y_spread_train, y_spread_test = y_spread[train_mask], y_spread[test_mask]
            
            logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
            
            # Create datasets
            train_dataset_wl = NBADataset(X_train, y_win_train)
            test_dataset_wl = NBADataset(X_test, y_win_test, 
                                       scaler=train_dataset_wl.scaler, fit_scaler=False)
            
            train_dataset_spread = NBADataset(X_train, y_spread_train)
            test_dataset_spread = NBADataset(X_test, y_spread_test,
                                           scaler=train_dataset_spread.scaler, fit_scaler=False)
            
            # Create data loaders
            train_loader_wl = DataLoader(train_dataset_wl, batch_size=64, shuffle=True)
            test_loader_wl = DataLoader(test_dataset_wl, batch_size=64, shuffle=False)
            
            train_loader_spread = DataLoader(train_dataset_spread, batch_size=64, shuffle=True)
            test_loader_spread = DataLoader(test_dataset_spread, batch_size=64, shuffle=False)
            
            # Train Win/Loss model
            logger.info("Training Win/Loss model...")
            win_loss_model = WinLossModel(input_size=X.shape[1])
            criterion_wl = nn.BCELoss()
            optimizer_wl = optim.Adam(win_loss_model.parameters(), lr=0.001, weight_decay=1e-5)
            
            win_loss_model, _, _ = self.train_model(
                win_loss_model, train_loader_wl, test_loader_wl, 
                criterion_wl, optimizer_wl
            )
            
            # Train Spread model
            logger.info("Training Spread model...")
            spread_model = SpreadModel(input_size=X.shape[1])
            criterion_spread = nn.MSELoss()
            optimizer_spread = optim.Adam(spread_model.parameters(), lr=0.001, weight_decay=1e-5)
            
            spread_model, _, _ = self.train_model(
                spread_model, train_loader_spread, test_loader_spread,
                criterion_spread, optimizer_spread
            )
            
            # Evaluate models
            logger.info("Evaluating models...")
            wl_metrics, _, _ = self.evaluate_model(win_loss_model, test_loader_wl, True)
            spread_metrics, _, _ = self.evaluate_model(spread_model, test_loader_spread, False)
            
            logger.info(f"Win/Loss metrics: {wl_metrics}")
            logger.info(f"Spread metrics: {spread_metrics}")
            
            # Save models
            model_data = {
                'win_loss_model_state': win_loss_model.state_dict(),
                'spread_model_state': spread_model.state_dict(),
                'win_loss_scaler': train_dataset_wl.scaler,
                'spread_scaler': train_dataset_spread.scaler,
                'feature_columns': feature_columns,
                'model_performance': {
                    'win_loss': wl_metrics,
                    'spread': spread_metrics
                },
                'training_date': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            if len(X_test) < 50:
                logger.warning("Very small test set - results may not be reliable")
            
            # Save with versioning
            model_path = self.save_model_artifacts(model_data)
            model_data['model_path'] = model_path
            
            logger.info("Model training completed successfully!")
            return model_data

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise


class NBAPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path is None:
            model_path = Path("models") / "nba_models_latest.pth"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model_data = torch.load(model_path, map_location=self.device)
            self._initialize_models()
            logger.info(f"Models loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise


    def _initialize_models(self):
        """Initialize models with proper error handling"""
        try:
            feature_count = len(self.model_data['feature_columns'])
            
            self.win_loss_model = WinLossModel(feature_count)
            self.spread_model = SpreadModel(feature_count)

            self.win_loss_model.load_state_dict(self.model_data['win_loss_model_state'])
            self.spread_model.load_state_dict(self.model_data['spread_model_state'])

            self.win_loss_model.to(self.device)
            self.spread_model.to(self.device)

            self.win_loss_model.eval()
            self.spread_model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise


    def predict_game(self, features_dict):
        """Enhanced prediction with validation"""
        try:
            # Validate input features
            missing_features = set(self.model_data['feature_columns']) - set(features_dict.keys())
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    features_dict[feature] = 0

            # Create feature array
            feature_array = np.array([[
                features_dict.get(col, 0) for col in self.model_data['feature_columns']
            ]])

            # Make predictions
            with torch.no_grad():
                wl_features = torch.FloatTensor(
                    self.model_data['win_loss_scaler'].transform(feature_array)
                ).to(self.device)

                spread_features = torch.FloatTensor(
                    self.model_data['spread_scaler'].transform(feature_array)
                ).to(self.device)

                win_prob = self.win_loss_model(wl_features).cpu().numpy()[0, 0]
                spread_pred = self.spread_model(spread_features).cpu().numpy()[0, 0]

            # Validate predictions
            win_prob = max(0.0, min(1.0, float(win_prob)))  # Clamp to [0,1]
            confidence = abs(win_prob - 0.5) * 2

            return {
                'home_win_probability': win_prob,
                'predicted_spread': float(spread_pred),
                'confidence': confidence,
                'prediction_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
