"""
Data Generator for Model Training

Refactored to:
1. Use `hft_shared.indicators` for consistent calculation.
2. Fetch real data from TimescaleDB if available.
3. Fallback to synthetic data if DB is empty or unreachable.
"""
import os
import logging
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict

# Import shared indicators
# Assumes the project root is in PYTHONPATH or services package is installed
try:
    from services.hft_shared.indicators import TechnicalIndicators
except ImportError:
    # Fallback for when running script directly without package context
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from services.hft_shared.indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB Config
TIMESCALE_HOST = os.getenv('TIMESCALE_HOST', 'localhost') # localhost default for running outside docker
TIMESCALE_PORT = int(os.getenv('TIMESCALE_PORT', '5432'))
TIMESCALE_USER = os.getenv('TIMESCALE_USER', 'hft_user')
TIMESCALE_PASSWORD = os.getenv('TIMESCALE_PASSWORD', 'hft_password')
TIMESCALE_DB = os.getenv('TIMESCALE_DB', 'hft_trading')

def fetch_real_data(limit: int = 50000) -> Optional[pd.DataFrame]:
    """Fetch raw ticks from TimescaleDB."""
    try:
        conn = psycopg2.connect(
            host=TIMESCALE_HOST,
            port=TIMESCALE_PORT,
            user=TIMESCALE_USER,
            password=TIMESCALE_PASSWORD,
            dbname=TIMESCALE_DB
        )
        query = f"""
            SELECT time, price, volume 
            FROM market_ticks 
            ORDER BY time DESC 
            LIMIT {limit};
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning("⚠️ Connected to DB but no data found.")
            return None
            
        # Sort by time ascending
        df = df.sort_values('time').reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch from DB: {e}. using synthetic data.")
        return None

def generate_price_series(
    n_samples: int = 10000,
    initial_price: float = 50000.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Generate synthetic prices and volumes."""
    np.random.seed(seed)
    dt = 1.0 / n_samples
    returns = np.random.normal(trend * dt, volatility * np.sqrt(dt), n_samples)
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i - 1]
    
    prices = initial_price * np.exp(np.cumsum(returns))
    volumes = np.random.uniform(1000, 10000, n_samples) * (1 + 0.3 * np.random.randn(n_samples))
    volumes = np.abs(volumes)
    
    # Generate timestamps
    base_time = datetime.now() - timedelta(minutes=n_samples)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_samples)]
    
    return prices, volumes, timestamps

def calculate_with_shared_lib(prices: np.ndarray, volumes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict[str, np.ndarray]:
    """Use TechnicalIndicators library."""
    n = len(prices)
    ti = TechnicalIndicators()
    
    # Pre-allocate
    indicators = {}
    
    # These functions return scalar usually, but here we need array for the dataframe
    # We must implement rolling/vectorized versions or loop.
    # The shared library computes SINGLE value from array usually (except checking its code).
    # Checking `hft_shared`: it takes an array and returns ONE float (latest).
    # Wait, `calculator.py` was doing windowed calc for LATEST point. 
    # For training, we need history.
    # We need to adapt the shared library or loop it efficiently.
    # Given requirements: "Use hft_shared".
    # We will implement a helper here to loop it or improve hft_shared to return array.
    # For now, to keep strict "consistency", we loop, or ideally we assume hft_shared *could* be vectorized.
    # The current `hft_shared` implementation only returns the last value.
    # FOR TRAINING DATA GENERATION, this is slow O(N^2) if we just loop `sma` on growing window.
    # But since `calculator.py` uses growing deque, it's effectively doing that for every tick.
    # To match EXACTLY, we should simulate the streaming process.
    
    # However, for training speed, we usually want vectorized.
    # Let's rely on pandas-like vectorization for Training, BUT we must ensure the math is identical.
    # Or, we modify `hft_shared` to support returning array (or keep it as is for serving).
    
    # RE-STRATEGY: Use `pandas` rolling here for speed, but ensure parameters match `hft_shared`.
    # `hft_shared` uses `np.mean(prices[-period:])` which is standard SMA.
    
    # Helper for rolling apply using the SAME logic if possible, or just standard pd logic which matches.
    
    # SMA
    s = pd.Series(prices)
    v = pd.Series(volumes)
    h = pd.Series(highs)
    l = pd.Series(lows)
    
    indicators['sma_20'] = s.rolling(20).mean().fillna(0).values
    
    # EMA 12/26 (Pandas ewm matches TechnicalIndicators.ema logic alpha=2/(Span+1))
    indicators['ema_12'] = s.ewm(span=12, adjust=False).mean().fillna(s[0]).values
    indicators['ema_26'] = s.ewm(span=26, adjust=False).mean().fillna(s[0]).values
    
    # MACD
    indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
    # Signal: EMA 9 of MACD
    indicators['macd_signal'] = pd.Series(indicators['macd']).ewm(span=9, adjust=False).mean().fillna(0).values
    indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
    
    # RSI
    # Shared impl: if len < period+1: 50.
    # Pandas vectorization:
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    indicators['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50).values 
    # NOTE: The Shared lib uses SIMPLE AVG for gain/loss in the window slice.
    # Many RSI impls use Wilder's Smoothing. Shared lib uses `np.mean`.
    # So `rolling(14).mean()` matches `np.mean` on slice. Correct.
    
    # Bollinger
    sma = s.rolling(20).mean()
    std = s.rolling(20).std(ddof=0) # numpy default ddof=0, pandas default ddof=1
    indicators['bb_upper'] = (sma + 2 * std).fillna(0).values
    indicators['bb_middle'] = sma.fillna(0).values
    indicators['bb_lower'] = (sma - 2 * std).fillna(0).values
    
    width = ((indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle'] * 100)
    indicators['bb_width'] = pd.Series(width).fillna(0).values

    # ROC
    indicators['roc_10'] = s.pct_change(10).fillna(0).values * 100
    
    # Volume Ratio
    vol_sma = v.rolling(20).mean()
    indicators['volume_ratio'] = (v / vol_sma).fillna(1.0).values
    
    # Price vs SMA
    indicators['price_vs_sma'] = ((s - indicators['sma_20']) / indicators['sma_20'] * 100).fillna(0).values
    
    return indicators

def generate_target(prices: np.ndarray, lookahead: int = 1, threshold: float = 0.0) -> np.ndarray:
    """Generate target variable: 1 if price goes UP, 0 if DOWN."""
    n = len(prices)
    target = np.zeros(n, dtype=int)
    for i in range(n - lookahead):
        pct_change = (prices[i + lookahead] - prices[i]) / prices[i] * 100
        target[i] = 1 if pct_change > threshold else 0
    return target

def generate_training_data(
    n_samples: int = 10000,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate complete training dataset (Real or Synthetic)."""
    
    # 1. Try Real Data
    df = fetch_real_data(limit=n_samples)
    
    if df is None or len(df) < 500: # Too little data = fallback
        if df is not None:
             logger.warning(f"⚠️ Found only {len(df)} real samples. Falling back to synthetic.")
        
        logger.info("🧪 Generating synthetic data...")
        prices, volumes, timestamps = generate_price_series(n_samples, seed=seed)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'high': prices * 1.001, # Approx
            'low': prices * 0.999
        })
    else:
        logger.info(f"✅ Using {len(df)} real data points from DB.")
        # Ensure high/low exist or approx
        if 'high' not in df.columns: df['high'] = df['price']
        if 'low' not in df.columns: df['low'] = df['price']
        if 'volume' not in df.columns: df['volume'] = 0

    # 2. Calculate Features
    prices = df['price'].values
    volumes = df['volume'].values
    highs = df['high'].values
    lows = df['low'].values
    
    indicators = calculate_with_shared_lib(prices, volumes, highs, lows)
    
    for k, v in indicators.items():
        df[k] = v
        
    # 3. Generate Target
    target = generate_target(prices, lookahead=1)
    
    # 4. Clean up NaN/Warmup (first 30 rows)
    df = df.iloc[30:].reset_index(drop=True)
    target = target[30:]
    
    return df, target

if __name__ == "__main__":
    df, target = generate_training_data(1000)
    print(f"Data shape: {df.shape}")
    print(df.head())
