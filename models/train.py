"""
Model Training Pipeline

Trains an XGBoost classifier to predict price direction (UP/DOWN)
and logs everything to MLflow for experiment tracking and model registry.

This script runs automatically and registers the best model.
"""
import os
import sys
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import optuna
from optuna.integration.mlflow import MLflowCallback

from data_generator import generate_training_data

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
EXPERIMENT_NAME = "hft-price-prediction"
MODEL_NAME = "hft-xgboost-classifier"

# Feature columns to use for training
FEATURE_COLUMNS = [
    'rsi_14',
    'macd',
    'macd_signal',
    'macd_histogram',
    'bb_width',
    'price_vs_sma',
    'roc_10',
    'volume_ratio',
    'spread_pct',
    'ema_12',
    'ema_26',
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix for training."""
    features = df[FEATURE_COLUMNS].copy()
    
    # Handle any NaN/inf values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    return features


def train_model(X_train, y_train, X_val, y_val) -> XGBClassifier:
    """
    Train XGBoost classifier with CPU-optimized parameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
    
    Returns:
        Trained XGBClassifier
    """
    # CPU-optimized parameters
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',  # CPU-optimized
        'n_jobs': 2,  # Limit CPU usage
        'random_state': 42,
        'use_label_encoder': False,
    }
    
    model = XGBClassifier(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=20) -> dict:
    """
    Run Optuna optimization to find best hyperparameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of trials
        
    Returns:
        Best parameters dictionary
    """
    logger.info("🔍 Starting Optuna hyperparameter optimization...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'n_jobs': 2,
            'random_state': 42,
            'use_label_encoder': False,
        }
        
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=10
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"🏆 Best validation AUC: {study.best_value:.4f}")
    logger.info(f"   Best params: {study.best_params}")
    
    # Return best params combined with fixed params
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist', # Ensure CPU optimized
        'n_jobs': 2,
        'random_state': 42,
        'use_label_encoder': False,
    })
    
    return best_params


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
    
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }
    
    return metrics


def get_feature_importance(model, feature_names: list) -> dict:
    """Get feature importance from trained model."""
    importance = model.feature_importances_
    return dict(zip(feature_names, importance.tolist()))


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 Starting HFT Model Training Pipeline")
    logger.info("=" * 60)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"📡 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Create or get experiment
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            logger.info(f"📁 Created new experiment: {EXPERIMENT_NAME}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"📁 Using existing experiment: {EXPERIMENT_NAME}")
    except Exception as e:
        logger.error(f"❌ Failed to setup experiment: {e}")
        experiment_id = "0"
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Generate training data
    logger.info("📊 Generating training data...")
    df, target = generate_training_data(n_samples=15000, seed=42)
    logger.info(f"   Generated {len(df)} samples")
    logger.info(f"   Target distribution: UP={sum(target)} ({sum(target)/len(target)*100:.1f}%), DOWN={len(target)-sum(target)}")
    
    # Prepare features
    X = prepare_features(df)
    y = target
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    logger.info(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        
        # Optimize hyperparameters
        logger.info("⚡ Optimizing hyperparameters...")
        best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=10)
        
        # Log best params
        mlflow.log_params(best_params)
        
        # Train final model with best params
        logger.info("🏋️ Training final XGBoost classifier...")
        model = XGBClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=10
        )
        logger.info("   Training complete!")
        
        # Evaluate model
        logger.info("📈 Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            logger.info(f"   {name}: {value:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())
        logger.info(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Feature importance
        importance = get_feature_importance(model, FEATURE_COLUMNS)
        logger.info("📊 Feature Importance:")
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {feat}: {imp:.4f}")
            mlflow.log_metric(f"importance_{feat}", imp)
        
        # Log model
        logger.info("💾 Logging model to MLflow...")
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        
        # Get run info
        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ Model logged with run_id: {run_id}")
        
    logger.info("=" * 60)
    logger.info("🎉 Training pipeline completed successfully!")
    logger.info(f"   Model registered as: {MODEL_NAME}")
    logger.info(f"   View in MLflow UI: {MLFLOW_TRACKING_URI}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
