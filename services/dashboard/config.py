"""
Dashboard Configuration

Single source of truth for all dashboard settings.
"""

# API Endpoints
MODEL_SERVER_URL = "http://model-server:8001"
TRADING_BOT_URL = "http://trading-bot:8002"
REDIS_HOST = "redis"
REDIS_PORT = 6379

# Refresh Intervals (seconds)
REFRESH_KPI = 5          # KPI cards
REFRESH_CHART = 10       # Charts
REFRESH_TRADES = 3       # Trade table
REFRESH_RECOMMENDATIONS = 60  # Model recommendations

# Trading Settings
SYMBOL = "BTC/USDT"
CONFIDENCE_THRESHOLD = 0.55

# Chart Settings
CHART_HEIGHT = 300
MAX_TRADES_DISPLAY = 20
PNL_HISTORY_POINTS = 50

# Thresholds for recommendations
THRESHOLDS = {
    "accuracy_low": 0.52,
    "accuracy_medium": 0.55,
    "confidence_low": 0.55,
    "drawdown_high": 0.10,  # 10%
    "win_rate_low": 0.45,
    "latency_high": 50,  # ms
}
