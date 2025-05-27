import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, brier_score_loss, mean_absolute_error, 
    mean_squared_error, calibration_curve
)
from sklearn.calibration import CalibratedClassifierCV
import logging
import os
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery, storage
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

logger = logging.getLogger(__name__)

class NBADataset(Dataset):
    """Dataset with preprocessing and sample weighting"""
    
    def __init__(self, features, targets, scaler=None, fit_scaler=True, sample_weights=None):
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
            
        # Handle NaN values before scaling
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if fit_scaler:
            self.features = torch.FloatTensor(self.scaler.fit_transform(features))
        else:
            self.features = torch.FloatTensor(self.scaler.transform(features))
            
        self.targets = torch.FloatTensor(targets)
        self.sample_weights = torch.FloatTensor(sample_weights) if sample_weights is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.sample_weights is not None:
            return self.features[idx], self.targets[idx], self.sample_weights[idx]
        return self.features[idx], self.targets[idx]

class WinLossModel(nn.Module):
    """Model architecture with regularization and calibration"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], 
                 dropout_rate=0.3, use_batch_norm=True):
        super(WinLossModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
                
            layers.append(nn.ReLU())
            
            # Graduated dropout - higher dropout in earlier layers
            current_dropout = dropout_rate * (0.8 ** i) if i < len(hidden_sizes) - 1 else dropout_rate / 2
            layers.append(nn.Dropout(current_dropout))
            
            prev_size = hidden_size
        
        # Final prediction layers with temperature scaling capability
        layers.extend([
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32) if use_batch_norm else nn.Identity(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
            # Note: No sigmoid here - using BCEWithLogitsLoss for numerical stability
        ])
        
        self.network = nn.Sequential(*layers)
        self.temperature = nn.Parameter(torch.ones(1))  # For temperature scaling
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        logits = self.network(x)
        return logits / self.temperature  # Temperature scaling for calibration
    
    def predict_proba(self, x):
        """Get calibrated probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

class SpreadModel(nn.Module):
    """Spread prediction model with uncertainty estimation"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=0.3):
        super(SpreadModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate * (0.8 ** i))
            ])
            prev_size = hidden_size
        
        # Final layers - predict both mean and log variance for uncertainty
        layers.extend([
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1)
        ])
        
        self.feature_layers = nn.Sequential(*layers)
        self.mean_head = nn.Linear(32, 1)
        self.log_var_head = nn.Linear(32, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_layers(x)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        return mean, log_var
    
    def predict_with_uncertainty(self, x):
        """Get predictions with uncertainty estimates"""
        with torch.no_grad():
            mean, log_var = self.forward(x)
            std = torch.exp(0.5 * log_var)
            return mean, std

class ModelCalibrator:
    """Model calibration utilities"""
    
    def __init__(self):
        self.calibrator = None
        self.is_fitted = False
    
    def fit_temperature_scaling(self, model, val_loader, device):
        """Fit temperature scaling for better calibration"""
        model.eval()
        
        # Collect validation predictions and targets
        logits_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    features, targets, _ = batch
                else:
                    features, targets = batch
                
                features = features.to(device)
                
                # Get raw logits (before temperature scaling)
                original_temp = model.temperature.item()
                model.temperature.data = torch.ones(1).to(device)
                
                logits = model.network(features)
                logits_list.append(logits.cpu())
                targets_list.append(targets)
                
                # Restore temperature
                model.temperature.data = torch.tensor([original_temp]).to(device)
        
        logits = torch.cat(logits_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        
        # Optimize temperature
        temperature = nn.Parameter(torch.ones(1))
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = nn.BCEWithLogitsLoss()(scaled_logits, targets.unsqueeze(1))
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Update model temperature
        model.temperature.data = temperature.data.to(device)
        logger.info(f"Optimized temperature: {temperature.item():.3f}")
        
        return temperature.item()
    
    def evaluate_calibration(self, y_true, y_prob, n_bins=10):
        """Evaluate model calibration"""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return {
                'expected_calibration_error': ece,
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
                'is_well_calibrated': ece < 0.1  # Threshold for good calibration
            }
            
        except Exception as e:
            logger.warning(f"Calibration evaluation failed: {e}")
            return {'expected_calibration_error': None, 'is_well_calibrated': False}

class AdvancedNBAModelTrainer:
    """Model trainer with comprehensive evaluation"""
    
    def __init__(self, bigquery_client):
        self.client = bigquery_client
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize GCS client for model storage
        try:
            self.gcs_client = storage.Client()
            self.bucket_name = settings.GCS_BUCKET
        except Exception as e:
            logger.warning(f"GCS client initialization failed: {e}")
            self.gcs_client = None
        
        self.calibrator = ModelCalibrator()
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model directory: {self.model_dir}")

    def load_training_data(self, min_date=None, feature_version="advanced_v1.0"):
        """Load training data with filtering and validation"""
        if min_date is None:
            min_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT *
        FROM `{settings.PROJECT_ID}.{settings.DATASET_ID}.feature_set`
        WHERE game_date >= '{min_date}'
        AND wl_home IS NOT NULL
        AND home_team_score IS NOT NULL
        AND away_team_score IS NOT NULL
        AND feature_engineering_version = '{feature_version}'
        ORDER BY game_date
        """
        
        logger.info(f"Loading training data from {min_date}...")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} records")
        
        if df.empty:
            raise ValueError("No training data found with specified criteria")
        
        # Data quality checks
        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        logger.info(f"Overall null percentage: {null_pct:.2%}")
        
        return df

    def prepare_features(self, df):
        """Feature preparation with automatic feature selection"""
        
        # Define comprehensive feature categories
        feature_patterns = [
            'home_team_points_avg_L', 'away_team_points_avg_L',
            'home_team_rebounds_avg_L', 'away_team_rebounds_avg_L', 
            'home_team_assists_avg_L', 'away_team_assists_avg_L',
            'home_team_fg_pct_avg_L', 'away_team_fg_pct_avg_L',
            'home_team_3p_pct_avg_L', 'away_team_3p_pct_avg_L',
            'home_team_ts_pct_avg_L', 'away_team_ts_pct_avg_L',
            'home_team_efg_pct_avg_L', 'away_team_efg_pct_avg_L',
            'home_offensive_rating_est_avg_L', 'away_offensive_rating_est_avg_L',
            'home_avg_off_rating_avg_L', 'away_avg_off_rating_avg_L',
            'home_avg_def_rating_avg_L', 'away_avg_def_rating_avg_L',
            'home_avg_pace_avg_L', 'away_avg_pace_avg_L',
            'diff_', 'home_rest_days', 'away_rest_days', 'rest_advantage',
            'home_back_to_back', 'away_back_to_back', 'rest_mismatch',
            'home_h2h_win_pct', 'home_h2h_avg_margin', 'days_since_last_h2h',
            'h2h_games_played', 'margin_volatility',
            'day_of_week', 'month', 'is_weekend', 'season_progress'
        ]
        
        # Find matching columns
        feature_columns = []
        for pattern in feature_patterns:
            matching_cols = [col for col in df.columns if pattern in col]
            feature_columns.extend(matching_cols)
        
        # Remove duplicates and validate existence
        feature_columns = list(set(feature_columns))
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Feature quality filtering
        valid_features = []
        for feature in feature_columns:
            # Check null percentage
            null_pct = df[feature].isnull().sum() / len(df)
            if null_pct > 0.5:
                logger.warning(f"Removing {feature}: {null_pct:.1%} null values")
                continue
            
            # Check variance
            if df[feature].nunique() <= 1:
                logger.warning(f"Removing {feature}: no variance")
                continue
            
            # Check for infinite values
            if np.isinf(df[feature]).any():
                logger.warning(f"Removing {feature}: contains infinite values")
                continue
                
            valid_features.append(feature)
        
        logger.info(f"Selected {len(valid_features)} valid features from {len(feature_columns)} candidates")
        return valid_features

    def create_temporal_split(self, df, train_ratio=0.7, val_ratio=0.15):
        """Create time-based splits to prevent data leakage"""
        df_sorted = df.sort_values('game_date')
        
        n_total = len(df_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = df_sorted.iloc[:n_train]
        val_df = df_sorted.iloc[n_train:n_train + n_val]
        test_df = df_sorted.iloc[n_train + n_val:]
        
        logger.info(f"Temporal split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Train period: {train_df['game_date'].min()} to {train_df['game_date'].max()}")
        logger.info(f"Val period: {val_df['game_date'].min()} to {val_df['game_date'].max()}")
        logger.info(f"Test period: {test_df['game_date'].min()} to {test_df['game_date'].max()}")
        
        return train_df, val_df, test_df

    def create_balanced_sampler(self, targets):
        """Create weighted sampler for handling class imbalance"""
        unique_targets = np.unique(targets)
        class_counts = np.bincount(targets.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[targets.astype(int)]
        
        logger.info(f"Class distribution: {dict(zip(unique_targets, class_counts))}")
        logger.info(f"Class weights: {dict(zip(unique_targets, class_weights))}")
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def train_model_with_validation(self, model, train_loader, val_loader, 
                                   criterion, optimizer, scheduler=None,
                                   num_epochs=200, patience=20):
        """Training with monitoring"""
        
        model = model.to(self.device)
        
        # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        best_val_metric = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_batches = 0
            
            for batch in train_loader:
                if len(batch) == 3:  # With sample weights
                    features, targets, weights = batch
                    features, targets, weights = features.to(self.device), targets.to(self.device), weights.to(self.device)
                else:
                    features, targets = batch
                    features, targets = features.to(self.device), targets.to(self.device)
                    weights = None
                
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)
                
                optimizer.zero_grad()
                
                # Handle different model types
                if hasattr(model, 'predict_with_uncertainty'):
                    # Spread model with uncertainty
                    mean, log_var = model(features)
                    loss = self._gaussian_nll_loss(mean, log_var, targets)
                else:
                    # Classification model
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                
                # Apply sample weights if available
                if weights is not None and not hasattr(model, 'predict_with_uncertainty'):
                    loss = (loss * weights.unsqueeze(1)).mean()
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
            
            # Validation phase
            val_loss, val_metric = self._validate_model(model, val_loader, criterion)
            
            # Record history
            avg_train_loss = train_loss / train_batches
            history['train_losses'].append(avg_train_loss)
            history['val_losses'].append(val_loss)
            history['val_metrics'].append(val_metric)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Model improvement check
            if hasattr(model, 'predict_with_uncertainty'):
                # For regression, use loss
                metric_improved = val_loss < best_val_loss
                if metric_improved:
                    best_val_loss = val_loss
            else:
                # For classification, use AUC
                metric_improved = val_metric > best_val_metric
                if metric_improved:
                    best_val_metric = val_metric
                    best_val_loss = val_loss
            
            # Early stopping and best model saving
            if metric_improved:
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Logging
            if epoch % 20 == 0 or patience_counter >= patience:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}, '
                          f'LR: {current_lr:.6f}, Patience: {patience_counter}')
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with metric: {best_val_metric:.4f}")
        
        return model, history

    def _gaussian_nll_loss(self, mean, log_var, targets):
        """Gaussian negative log-likelihood loss for uncertainty estimation"""
        var = torch.exp(log_var)
        loss = 0.5 * (log_var + torch.pow(targets - mean, 2) / var)
        return loss.mean()

    def _validate_model(self, model, val_loader, criterion):
        """Validate model and return loss and primary metric"""
        model.eval()
        val_loss = 0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    features, targets, _ = batch
                else:
                    features, targets = batch
                
                features, targets = features.to(self.device), targets.to(self.device)
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)
                
                if hasattr(model, 'predict_with_uncertainty'):
                    # Spread model
                    mean, log_var = model(features)
                    loss = self._gaussian_nll_loss(mean, log_var, targets)
                    predictions.extend(mean.cpu().numpy().flatten())
                else:
                    # Classification model
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    predictions.extend(probs)
                
                val_loss += loss.item()
                targets_list.extend(targets.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate primary metric
        if hasattr(model, 'predict_with_uncertainty'):
            # For regression: negative MAE as metric (higher is better)
            val_metric = -mean_absolute_error(targets_list, predictions)
        else:
            # For classification: AUC
            try:
                val_metric = roc_auc_score(targets_list, predictions)
            except:
                val_metric = 0.5  # Random baseline
        
        return avg_val_loss, val_metric

    def evaluate_model_comprehensive(self, model, test_loader, is_classification=True):
        """Comprehensive model evaluation with calibration analysis"""
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    features, targets, _ = batch
                else:
                    features, targets = batch
                
                features = features.to(self.device)
                
                if hasattr(model, 'predict_with_uncertainty'):
                    # Spread model with uncertainty
                    mean, std = model.predict_with_uncertainty(features)
                    predictions = mean.cpu().numpy().flatten()
                    uncertainties = std.cpu().numpy().flatten()
                    all_predictions.extend(predictions)
                    all_uncertainties.extend(uncertainties)
                else:
                    # Classification model
                    probabilities = model.predict_proba(features).cpu().numpy().flatten()
                    predictions = (probabilities > 0.5).astype(int)
                    all_probabilities.extend(probabilities)
                    all_predictions.extend(predictions)
                
                all_targets.extend(targets.numpy())
        
        if is_classification:
            # Classification metrics
            metrics = {
                'accuracy': accuracy_score(all_targets, all_predictions),
                'precision': precision_score(all_targets, all_predictions, zero_division=0),
                'recall': recall_score(all_targets, all_predictions, zero_division=0),
                'f1': f1_score(all_targets, all_predictions, zero_division=0),
                'auc': roc_auc_score(all_targets, all_probabilities),
                'log_loss': log_loss(all_targets, all_probabilities),
                'brier_score': brier_score_loss(all_targets, all_probabilities)
            }
            
            # Calibration analysis
            calibration_results = self.calibrator.evaluate_calibration(
                np.array(all_targets), np.array(all_probabilities)
            )
            metrics.update(calibration_results)
            
        else:
            # Regression metrics
            mae = mean_absolute_error(all_targets, all_predictions)
            mse = mean_squared_error(all_targets, all_predictions)
            rmse = np.sqrt(mse)
            
            metrics = {
                'mae': mae,
                'mse': mse, 
                'rmse': rmse,
                'mean_target': np.mean(all_targets),
                'std_target': np.std(all_targets)
            }
            
            # Uncertainty metrics if available
            if all_uncertainties:
                metrics['mean_uncertainty'] = np.mean(all_uncertainties)
                metrics['uncertainty_std'] = np.std(all_uncertainties)
        
        return metrics, all_predictions, all_targets, all_probabilities if is_classification else all_uncertainties

    def save_model_artifacts_to_gcs(self, model_data):
        """Save model artifacts to GCS with comprehensive versioning"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Local save first
            local_path = self.model_dir / f"nba_models_{timestamp}.pth"
            torch.save(model_data, local_path)
            
            if self.gcs_client:
                # Upload to GCS
                bucket = self.gcs_client.bucket(self.bucket_name)
                
                # Upload main model file
                blob_name = f"models/nba_models_{timestamp}.pth"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(local_path))
                
                # Update latest symlink
                latest_blob = bucket.blob("models/nba_models_latest.pth")
                latest_blob.upload_from_filename(str(local_path))
                
                # Save comprehensive metadata
                metadata = {
                    'timestamp': model_data['training_date'],
                    'performance': model_data['model_performance'],
                    'feature_count': len(model_data['feature_columns']),
                    'training_samples': model_data['training_samples'],
                    'test_samples': model_data['test_samples'],
                    'model_version': model_data['model_version'],
                    'calibration_info': model_data.get('calibration_info', {}),
                    'training_curves': {
                        key: value for key, value in model_data.get('training_curves', {}).items()
                        if not isinstance(value, list) or len(value) < 1000  # Limit large arrays
                    }
                }
                
                metadata_blob = bucket.blob(f"models/metadata_{timestamp}.json")
                metadata_blob.upload_from_string(
                    json.dumps(metadata, indent=2, default=str),
                    content_type='application/json'
                )
                
                logger.info(f"Model artifacts saved to GCS: {blob_name}")
                return f"gs://{self.bucket_name}/{blob_name}"
            else:
                logger.warning("GCS client not available, saving locally only")
                return str(local_path)
                
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            return str(local_path)

    def train_all_models(self):
        """Train comprehensive model suite with evaluation"""
        try:
            start_time = datetime.now()
            
            # Load and prepare data
            df = self.load_training_data()
            feature_columns = self.prepare_features(df)
            
            if not feature_columns:
                raise ValueError("No valid features found")
            
            logger.info(f"Training with {len(feature_columns)} features on {len(df)} samples")
            
            # Create temporal splits
            train_df, val_df, test_df = self.create_temporal_split(df)
            
            # Prepare features and targets
            X_train = train_df[feature_columns].fillna(0).values.astype(np.float32)
            X_val = val_df[feature_columns].fillna(0).values.astype(np.float32)
            X_test = test_df[feature_columns].fillna(0).values.astype(np.float32)
            
            y_win_train = train_df['wl_home'].map({'W': 1, 'L': 0}).values.astype(np.float32)
            y_win_val = val_df['wl_home'].map({'W': 1, 'L': 0}).values.astype(np.float32)
            y_win_test = test_df['wl_home'].map({'W': 1, 'L': 0}).values.astype(np.float32)
            
            y_spread_train = (train_df['home_team_score'] - train_df['away_team_score']).values.astype(np.float32)
            y_spread_val = (val_df['home_team_score'] - val_df['away_team_score']).values.astype(np.float32)
            y_spread_test = (test_df['home_team_score'] - test_df['away_team_score']).values.astype(np.float32)
            
            logger.info(f"Win rate in training: {y_win_train.mean():.3f}")
            logger.info(f"Average spread in training: {y_spread_train.mean():.2f}")
            
            # Create datasets
            train_dataset_wl = NBADataset(X_train, y_win_train)
            val_dataset_wl = NBADataset(X_val, y_win_val, scaler=train_dataset_wl.scaler, fit_scaler=False)
            test_dataset_wl = NBADataset(X_test, y_win_test, scaler=train_dataset_wl.scaler, fit_scaler=False)
            
            train_dataset_spread = NBADataset(X_train, y_spread_train)
            val_dataset_spread = NBADataset(X_val, y_spread_val, scaler=train_dataset_spread.scaler, fit_scaler=False)
            test_dataset_spread = NBADataset(X_test, y_spread_test, scaler=train_dataset_spread.scaler, fit_scaler=False)
            
            # Create data loaders
            wl_sampler = self.create_balanced_sampler(y_win_train)
            
            train_loader_wl = DataLoader(train_dataset_wl, batch_size=settings.BATCH_SIZE, sampler=wl_sampler)
            val_loader_wl = DataLoader(val_dataset_wl, batch_size=settings.BATCH_SIZE, shuffle=False)
            test_loader_wl = DataLoader(test_dataset_wl, batch_size=settings.BATCH_SIZE, shuffle=False)
            
            train_loader_spread = DataLoader(train_dataset_spread, batch_size=settings.BATCH_SIZE, shuffle=True)
            val_loader_spread = DataLoader(val_dataset_spread, batch_size=settings.BATCH_SIZE, shuffle=False)
            test_loader_spread = DataLoader(test_dataset_spread, batch_size=settings.BATCH_SIZE, shuffle=False)
            
            # Train Win/Loss model
            logger.info("Training Win/Loss model...")
            win_loss_model = WinLossModel(input_size=len(feature_columns))
            criterion_wl = nn.BCEWithLogitsLoss()
            optimizer_wl = optim.AdamW(win_loss_model.parameters(), 
                                     lr=settings.LEARNING_RATE, weight_decay=1e-4)
            scheduler_wl = optim.lr_scheduler.ReduceLROnPlateau(optimizer_wl, patience=10, factor=0.5)
            
            win_loss_model, wl_history = self.train_model_with_validation(
                win_loss_model, train_loader_wl, val_loader_wl, 
                criterion_wl, optimizer_wl, scheduler_wl,
                num_epochs=settings.MAX_EPOCHS, patience=settings.EARLY_STOPPING_PATIENCE
            )
            
            # Calibrate Win/Loss model
            logger.info("Calibrating Win/Loss model...")
            optimal_temp = self.calibrator.fit_temperature_scaling(
                win_loss_model, val_loader_wl, self.device
            )
            
            # Train Spread model
            logger.info("Training Spread model...")
            spread_model = SpreadModel(input_size=len(feature_columns))
            optimizer_spread = optim.AdamW(spread_model.parameters(), 
                                         lr=settings.LEARNING_RATE, weight_decay=1e-4)
            scheduler_spread = optim.lr_scheduler.ReduceLROnPlateau(optimizer_spread, patience=10, factor=0.5)
            
            spread_model, spread_history = self.train_model_with_validation(
                spread_model, train_loader_spread, val_loader_spread,
                None, optimizer_spread, scheduler_spread,  # No criterion needed for custom loss
                num_epochs=settings.MAX_EPOCHS, patience=settings.EARLY_STOPPING_PATIENCE
            )
            
            # Comprehensive evaluation
            logger.info("Performing comprehensive evaluation...")
            wl_metrics, wl_preds, wl_targets, wl_probs = self.evaluate_model_comprehensive(
                win_loss_model, test_loader_wl, True
            )
            spread_metrics, spread_preds, spread_targets, spread_uncertainties = self.evaluate_model_comprehensive(
                spread_model, test_loader_spread, False
            )
            
            # Log detailed results
            logger.info("=== WIN/LOSS MODEL RESULTS ===")
            for metric, value in wl_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
                else:
                    logger.info(f"{metric}: {value}")
            
            logger.info("=== SPREAD MODEL RESULTS ===")
            for metric, value in spread_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
                else:
                    logger.info(f"{metric}: {value}")
            
            # Training time
            training_time = datetime.now() - start_time
            logger.info(f"Total training time: {training_time}")
            
            # Save comprehensive model data
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
                'training_curves': {
                    'win_loss': wl_history,
                    'spread': spread_history
                },
                'calibration_info': {
                    'optimal_temperature': optimal_temp,
                    'is_calibrated': wl_metrics.get('is_well_calibrated', False)
                },
                'training_date': datetime.now().isoformat(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'training_time_seconds': training_time.total_seconds(),
                'model_version': 'v1.0',
                'settings_used': {
                    'learning_rate': settings.LEARNING_RATE,
                    'batch_size': settings.BATCH_SIZE,
                    'max_epochs': settings.MAX_EPOCHS,
                    'early_stopping_patience': settings.EARLY_STOPPING_PATIENCE
                }
            }
            
            # Save to GCS
            model_path = self.save_model_artifacts_to_gcs(model_data)
            model_data['model_path'] = model_path
            
            logger.info("Model training completed successfully!")
            return model_data

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

class NBAPredictor:
    """Predictor with uncertainty quantification and calibration"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path is None:
            model_path = Path("models") / "nba_models_latest.pth"
        elif model_path.startswith("gs://"):
            # Download from GCS
            model_path = self._download_from_gcs(model_path)
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model_data = torch.load(model_path, map_location=self.device)
            self._initialize_models()
            logger.info(f"Models loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _download_from_gcs(self, gcs_path: str) -> str:
        """Download model from GCS"""
        try:
            from google.cloud import storage
            client = storage.Client()
            
            # Parse GCS path
            bucket_name = gcs_path.split("/")[2]
            blob_name = "/".join(gcs_path.split("/")[3:])
            
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            local_path = Path("models") / "downloaded_model.pth"
            local_path.parent.mkdir(exist_ok=True)
            
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded model from {gcs_path} to {local_path}")
            
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Error downloading from GCS: {e}")
            raise

    def _initialize_models(self):
        """Initialize models with error handling"""
        try:
            feature_count = len(self.model_data['feature_columns'])
            
            # Initialize Win/Loss model
            self.win_loss_model = WinLossModel(feature_count)
            self.win_loss_model.load_state_dict(self.model_data['win_loss_model_state'])
            self.win_loss_model.to(self.device)
            self.win_loss_model.eval()
            
            # Initialize Spread model
            self.spread_model = SpreadModel(feature_count)
            self.spread_model.load_state_dict(self.model_data['spread_model_state'])
            self.spread_model.to(self.device)
            self.spread_model.eval()
            
            # Load scalers
            self.win_loss_scaler = self.model_data['win_loss_scaler']
            self.spread_scaler = self.model_data['spread_scaler']
            
            logger.info(f"Models initialized with {feature_count} features")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def predict_game(self, features_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prediction with uncertainty quantification and calibration"""
        try:
            # Validate and prepare features
            missing_features = set(self.model_data['feature_columns']) - set(features_dict.keys())
            if missing_features:
                logger.warning(f"Missing features: {list(missing_features)[:5]}...")
                for feature in missing_features:
                    features_dict[feature] = 0.0
            
            # Create feature array
            feature_array = np.array([[
                float(features_dict.get(col, 0.0)) for col in self.model_data['feature_columns']
            ]])
            
            # Handle NaN/inf values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Make predictions
            with torch.no_grad():
                # Win/Loss prediction with calibration
                wl_features = torch.FloatTensor(
                    self.win_loss_scaler.transform(feature_array)
                ).to(self.device)
                
                win_prob = self.win_loss_model.predict_proba(wl_features).cpu().numpy()[0, 0]
                
                # Spread prediction with uncertainty
                spread_features = torch.FloatTensor(
                    self.spread_scaler.transform(feature_array)
                ).to(self.device)
                
                spread_mean, spread_std = self.spread_model.predict_with_uncertainty(spread_features)
                spread_pred = spread_mean.cpu().numpy()[0, 0]
                spread_uncertainty = spread_std.cpu().numpy()[0, 0]
            
            # Validate and clean predictions
            win_prob = max(0.0, min(1.0, float(win_prob)))
            spread_pred = float(spread_pred)
            spread_uncertainty = max(0.0, float(spread_uncertainty))
            
            # Calculate confidence metrics
            win_confidence = abs(win_prob - 0.5) * 2  # Distance from 50%
            spread_confidence = max(0.0, 1.0 - spread_uncertainty / 10.0)  # Inverse of uncertainty
            
            # Overall prediction confidence
            overall_confidence = (win_confidence + spread_confidence) / 2
            
            # Determine prediction strength
            prediction_strength = "low"
            if overall_confidence > 0.7:
                prediction_strength = "high"
            elif overall_confidence > 0.4:
                prediction_strength = "medium"
            
            return {
                'home_win_probability': win_prob,
                'away_win_probability': 1.0 - win_prob,
                'predicted_spread': spread_pred,
                'spread_uncertainty': spread_uncertainty,
                'spread_confidence_interval': {
                    'lower': spread_pred - 1.96 * spread_uncertainty,
                    'upper': spread_pred + 1.96 * spread_uncertainty
                },
                'win_confidence': win_confidence,
                'spread_confidence': spread_confidence,
                'overall_confidence': overall_confidence,
                'prediction_strength': prediction_strength,
                'model_metadata': {
                    'model_version': self.model_data.get('model_version', 'unknown'),
                    'training_date': self.model_data.get('training_date', 'unknown'),
                    'feature_count': len(self.model_data['feature_columns']),
                    'is_calibrated': self.model_data.get('calibration_info', {}).get('is_calibrated', False)
                },
                'prediction_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

    def batch_predict(self, games_features: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Batch prediction for multiple games"""
        predictions = []
        
        for game_features in games_features:
            try:
                prediction = self.predict_game(game_features)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                predictions.append(None)
        
        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'model_version': self.model_data.get('model_version', 'unknown'),
            'training_date': self.model_data.get('training_date', 'unknown'),
            'feature_count': len(self.model_data['feature_columns']),
            'training_samples': self.model_data.get('training_samples', 'unknown'),
            'test_samples': self.model_data.get('test_samples', 'unknown'),
            'performance': self.model_data.get('model_performance', {}),
            'calibration_info': self.model_data.get('calibration_info', {}),
            'feature_columns': self.model_data['feature_columns']
        }

if __name__ == "__main__":
    # Example usage and testing
    logger.info("NBA Models Module Loaded")
