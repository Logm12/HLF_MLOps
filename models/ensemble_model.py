"""
Ensemble Model - XGBoost + LightGBM + Random Forest Voting

Combines multiple models for more stable predictions.
"""
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from xgboost import XGBClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lstm_model import LSTMClassifier

import config

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Training metrics for a model."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_accuracy: float
    cv_std: float
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "roc_auc": round(self.roc_auc, 4),
            "cv_accuracy": round(self.cv_accuracy, 4),
            "cv_std": round(self.cv_std, 4),
        }


def create_xgboost_model() -> XGBClassifier:
    """Create XGBoost classifier with tuned parameters."""
    return XGBClassifier(
        **config.XGBOOST_PARAMS,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
    )


def create_lightgbm_model() -> LGBMClassifier:
    """Create LightGBM classifier with tuned parameters."""
    return LGBMClassifier(
        **config.LIGHTGBM_PARAMS,
        objective="binary",
    )


def create_random_forest_model() -> RandomForestClassifier:
    """Create Random Forest classifier with tuned parameters."""
    return RandomForestClassifier(**config.RF_PARAMS)


def create_ensemble_model() -> VotingClassifier:
    """
    Create ensemble voting classifier.
    
    Combines XGBoost, LightGBM, and Random Forest using soft voting
    (probability averaging).
    """
    estimators = [
        ("xgb", create_xgboost_model()),
        ("lgb", create_lightgbm_model()),
        ("rf", create_random_forest_model()),
        ("lstm", LSTMClassifier(input_dim=len(config.FEATURE_COLUMNS))),
    ]
    
    return VotingClassifier(
        estimators=estimators,
        voting="soft",  # Average probabilities
        n_jobs=2,
    )


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_full: Optional[pd.DataFrame] = None,
    y_full: Optional[pd.Series] = None,
    cv_folds: int = 5,
) -> ModelMetrics:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_full: Full dataset for CV (optional)
        y_full: Full labels for CV (optional)
        cv_folds: Number of CV folds
    
    Returns:
        ModelMetrics dataclass
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Cross-validation
    cv_accuracy = 0.0
    cv_std = 0.0
    if X_full is not None and y_full is not None:
        cv_scores = cross_val_score(model, X_full, y_full, cv=cv_folds, scoring="accuracy")
        cv_accuracy = cv_scores.mean()
        cv_std = cv_scores.std()
    
    return ModelMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        roc_auc=roc_auc,
        cv_accuracy=cv_accuracy,
        cv_std=cv_std,
    )


def get_feature_importance(
    model: VotingClassifier,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Get averaged feature importance from ensemble.
    
    Args:
        model: Trained VotingClassifier
        feature_names: List of feature names
    
    Returns:
        Dictionary of feature -> importance
    """
    importance_sum = np.zeros(len(feature_names))
    
    for name, estimator in model.named_estimators_.items():
        if hasattr(estimator, "feature_importances_"):
            importance_sum += estimator.feature_importances_
    
    # Average importance
    importance_avg = importance_sum / len(model.named_estimators_)
    
    # Sort by importance
    importance_dict = dict(zip(feature_names, importance_avg))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def prepare_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "target",
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple:
    """
    Prepare train/val/test splits.
    
    Uses time-based split to avoid lookahead bias.
    """
    # Remove rows with NaN target
    df = df.dropna(subset=[target_column])
    
    # Get features and target
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Time-based split (no shuffle to maintain temporal order)
    n = len(X)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    logger.info(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    logger.info(f"   Target distribution: {np.mean(y_train):.1%} UP in train")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Test ensemble creation
    model = create_ensemble_model()
    print(f"Ensemble model created: {model}")
    print(f"Estimators: {[name for name, _ in model.estimators]}")
