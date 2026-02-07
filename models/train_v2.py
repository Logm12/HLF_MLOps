"""
Training Pipeline V2 - Improved Model

Uses real data, advanced features, and ensemble model.
"""
import os
import sys
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

import config
from data_fetcher import fetch_and_save_data
from feature_engineering import compute_all_features, create_target
from ensemble_model import (
    create_ensemble_model,
    evaluate_model,
    get_feature_importance,
    prepare_data,
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Model Training Pipeline V2")
    logger.info("=" * 60)
    
    # ===========================================
    # 1. FETCH DATA
    # ===========================================
    logger.info("\n📊 Step 1: Fetching historical data...")
    df = fetch_and_save_data()
    
    if df.empty or len(df) < 1000:
        logger.error("❌ Insufficient data for training")
        return 1
    
    logger.info(f"   Loaded {len(df)} candles")
    
    # ===========================================
    # 2. FEATURE ENGINEERING
    # ===========================================
    logger.info("\n🔧 Step 2: Computing features...")
    df = compute_all_features(df)
    
    # ===========================================
    # 3. CREATE TARGET
    # ===========================================
    logger.info("\n🎯 Step 3: Creating target variable...")
    df["target"] = create_target(df, horizon=config.PREDICTION_HORIZON)
    
    # Remove last rows without target
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    
    logger.info(f"   Target distribution: {df['target'].mean():.1%} UP")
    
    # ===========================================
    # 4. PREPARE DATA SPLITS
    # ===========================================
    logger.info("\n📈 Step 4: Preparing data splits...")
    
    # Filter valid feature columns
    available_features = [f for f in config.FEATURE_COLUMNS if f in df.columns]
    logger.info(f"   Using {len(available_features)} features")
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        df,
        feature_columns=available_features,
        target_column="target",
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
    )
    
    # ===========================================
    # 5. TRAIN MODEL
    # ===========================================
    logger.info("\n🏋️ Step 5: Training ensemble model...")
    
    # Setup MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    
    try:
        mlflow.create_experiment(config.EXPERIMENT_NAME)
    except Exception:
        pass
    
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("data_points", len(df))
        mlflow.log_param("n_features", len(available_features))
        mlflow.log_param("prediction_horizon", config.PREDICTION_HORIZON)
        mlflow.log_param("timeframe", config.TIMEFRAME)
        mlflow.log_param("model_type", "ensemble")
        
        # Create and train model
        model = create_ensemble_model()
        
        # Combine train and val for fitting
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        logger.info("   Training XGBoost + LightGBM + Random Forest ensemble...")
        model.fit(X_train_full, y_train_full)
        logger.info("   ✅ Training complete!")
        
        # ===========================================
        # 6. EVALUATE MODEL
        # ===========================================
        logger.info("\n📈 Step 6: Evaluating model...")
        
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            X_full=X_train_full,
            y_full=y_train_full,
            cv_folds=5,
        )
        
        # Log metrics
        for name, value in metrics.to_dict().items():
            mlflow.log_metric(name, value)
            logger.info(f"   {name}: {value}")
        
        # ===========================================
        # 7. FEATURE IMPORTANCE
        # ===========================================
        logger.info("\n📊 Step 7: Feature importance...")
        
        importance = get_feature_importance(model, available_features)
        
        logger.info("   Top 10 features:")
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            logger.info(f"   {i+1}. {feat}: {imp:.4f}")
            mlflow.log_metric(f"importance_{feat}", imp)
        
        # ===========================================
        # 8. SAVE MODEL
        # ===========================================
        logger.info("\n💾 Step 8: Saving model to MLflow...")
        
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.MODEL_NAME,
        )
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"   ✅ Model logged with run_id: {run_id}")
    
    # ===========================================
    # SUMMARY
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Training Pipeline V2 completed!")
    logger.info("=" * 60)
    logger.info(f"   📊 Accuracy: {metrics.accuracy:.1%}")
    logger.info(f"   🎯 Precision: {metrics.precision:.1%}")
    logger.info(f"   📈 ROC AUC: {metrics.roc_auc:.1%}")
    logger.info(f"   🔄 CV Accuracy: {metrics.cv_accuracy:.1%} (+/- {metrics.cv_std*2:.1%})")
    logger.info(f"   📁 Model: {config.MODEL_NAME}")
    logger.info(f"   🔗 MLflow: {config.MLFLOW_TRACKING_URI}")
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
