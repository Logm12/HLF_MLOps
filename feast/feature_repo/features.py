"""
Feast Feature Definitions for HFT Trading System

This module defines entities and feature views for technical trading indicators.
Features are computed by the Feature Calculator and pushed to Redis for low-latency access.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, PushSource
from feast.types import Float64, Int64, String


# ==============================================
# ENTITIES
# ==============================================

# Trading symbol entity (e.g., BTC/USDT)
symbol_entity = Entity(
    name="symbol",
    join_keys=["symbol"],
    description="Trading pair symbol (e.g., BTC/USDT)",
)


# ==============================================
# PUSH SOURCE
# ==============================================

# Push source for real-time feature updates from Feature Calculator
trading_features_push_source = PushSource(
    name="trading_features_push",
    batch_source=None,  # Batch source not needed for real-time
)


# ==============================================
# FEATURE VIEW - Trading Features
# ==============================================

trading_features = FeatureView(
    name="trading_features",
    entities=[symbol_entity],
    ttl=timedelta(hours=1),  # Features expire after 1 hour
    schema=[
        # Price Features
        Field(name="price_current", dtype=Float64, description="Current price"),
        Field(name="price_change_pct", dtype=Float64, description="24h price change %"),
        Field(name="volume", dtype=Float64, description="24h trading volume"),
        Field(name="bid", dtype=Float64, description="Best bid price"),
        Field(name="ask", dtype=Float64, description="Best ask price"),
        Field(name="spread", dtype=Float64, description="Bid-Ask spread"),
        Field(name="spread_pct", dtype=Float64, description="Spread as % of price"),
        
        # Trend Indicators
        Field(name="sma_20", dtype=Float64, description="Simple Moving Average 20"),
        Field(name="ema_12", dtype=Float64, description="Exponential MA 12"),
        Field(name="ema_26", dtype=Float64, description="Exponential MA 26"),
        Field(name="price_vs_sma", dtype=Float64, description="Price position vs SMA %"),
        
        # Momentum Indicators
        Field(name="rsi_14", dtype=Float64, description="Relative Strength Index 14"),
        Field(name="macd", dtype=Float64, description="MACD line"),
        Field(name="macd_signal", dtype=Float64, description="MACD signal line"),
        Field(name="macd_histogram", dtype=Float64, description="MACD histogram"),
        Field(name="roc_10", dtype=Float64, description="Rate of Change 10 periods"),
        
        # Volatility Indicators
        Field(name="bb_upper", dtype=Float64, description="Bollinger upper band"),
        Field(name="bb_middle", dtype=Float64, description="Bollinger middle band"),
        Field(name="bb_lower", dtype=Float64, description="Bollinger lower band"),
        Field(name="bb_width", dtype=Float64, description="Bollinger band width %"),
        Field(name="atr_14", dtype=Float64, description="Average True Range 14"),
        
        # Volume Indicators
        Field(name="obv", dtype=Float64, description="On-Balance Volume"),
        Field(name="volume_sma", dtype=Float64, description="Volume SMA 20"),
        Field(name="volume_ratio", dtype=Float64, description="Volume vs SMA ratio"),
        
        # System Metrics
        Field(name="latency_ms", dtype=Float64, description="Feature calculation latency"),
        Field(name="data_age_ms", dtype=Float64, description="Age of source data"),
    ],
    source=trading_features_push_source,
    online=True,
    tags={"team": "hft", "version": "1.0"},
)


# ==============================================
# FEATURE VIEW - Model Performance Metrics
# ==============================================

model_metrics_push_source = PushSource(
    name="model_metrics_push",
    batch_source=None,
)

model_metrics = FeatureView(
    name="model_metrics",
    entities=[symbol_entity],
    ttl=timedelta(hours=24),
    schema=[
        # Prediction Metrics
        Field(name="last_prediction", dtype=Float64, description="Last model prediction"),
        Field(name="prediction_confidence", dtype=Float64, description="Prediction confidence"),
        Field(name="prediction_latency_ms", dtype=Float64, description="Model inference time"),
        
        # Accuracy Metrics (rolling window)
        Field(name="accuracy_1h", dtype=Float64, description="Accuracy last 1 hour"),
        Field(name="accuracy_24h", dtype=Float64, description="Accuracy last 24 hours"),
        Field(name="predictions_count", dtype=Int64, description="Total predictions made"),
        Field(name="correct_predictions", dtype=Int64, description="Correct predictions"),
        
        # Financial Performance
        Field(name="win_rate", dtype=Float64, description="Win rate %"),
        Field(name="profit_factor", dtype=Float64, description="Gross profit / Gross loss"),
        Field(name="sharpe_ratio", dtype=Float64, description="Risk-adjusted return"),
        Field(name="max_drawdown", dtype=Float64, description="Maximum drawdown %"),
        Field(name="total_pnl", dtype=Float64, description="Total profit/loss"),
        Field(name="trades_count", dtype=Int64, description="Total trades executed"),
    ],
    source=model_metrics_push_source,
    online=True,
    tags={"team": "hft", "version": "1.0"},
)
