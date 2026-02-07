"""
Configuration for Model Training V2

All settings in one place for easy tuning and testing.
"""

# ===========================================
# DATA SETTINGS
# ===========================================
SYMBOL = "BTCUSDT"
TIMEFRAME = "15m"  # 15-minute candles for better signal
LOOKBACK_DAYS = 730  # 2 years of data

# Binance API (public, no key needed)
BINANCE_BASE_URL = "https://api.binance.com"

# Data paths
DATA_DIR = "/app/data"
DATA_FILE = f"{DATA_DIR}/{SYMBOL.lower()}_{TIMEFRAME}.parquet"


# ===========================================
# FEATURE SETTINGS
# ===========================================
# Prediction horizon (in candles)
# 15min candles, 4 candles = 1 hour lookahead
PREDICTION_HORIZON = 4  # 1 hour ahead

# Technical indicator periods
INDICATOR_PERIODS = {
    "sma_short": 10,
    "sma_medium": 20,
    "sma_long": 50,
    "ema_short": 12,
    "ema_long": 26,
    "rsi": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2,
    "atr": 14,
    "adx": 14,
    "stoch": 14,
    "cci": 20,
    "mfi": 14,
}

# Feature columns for training
FEATURE_COLUMNS = [
    # Price-based
    "returns", "log_returns",
    # Trend
    "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
    "price_vs_sma20", "sma_cross",
    # Momentum
    "rsi", "rsi_category",
    "macd", "macd_signal", "macd_histogram",
    "stoch_k", "stoch_d",
    "cci", "mfi",
    # Volatility
    "bb_upper", "bb_lower", "bb_width", "bb_position",
    "atr", "atr_pct",
    # Trend strength
    "adx", "plus_di", "minus_di",
    # Volume
    "volume_sma", "volume_ratio",
    # Custom
    "hour", "day_of_week",
]


# ===========================================
# MODEL SETTINGS
# ===========================================
# Train/Val/Test split
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# XGBoost parameters
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": 42,
}

# LightGBM parameters
LIGHTGBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": 2,
    "random_state": 42,
    "verbose": -1,
}

# Random Forest parameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "n_jobs": 2,
    "random_state": 42,
}


# ===========================================
# MLFLOW SETTINGS
# ===========================================
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "hft-price-prediction-v2"
MODEL_NAME = "hft-ensemble-classifier"
