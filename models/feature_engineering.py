"""
Feature Engineering - Technical Indicators

Computes all technical indicators from OHLCV data.
Pure functions for easy testing and debugging.
"""
import numpy as np
import pandas as pd
from typing import Optional
import logging

import config

logger = logging.getLogger(__name__)


# ===========================================
# UTILITY FUNCTIONS
# ===========================================

def safe_divide(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Safely divide arrays, handling zeros."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b != 0, a / b, default)
    return np.nan_to_num(result, nan=default, posinf=default, neginf=default)


# ===========================================
# TREND INDICATORS
# ===========================================

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma_cross(short_sma: pd.Series, long_sma: pd.Series) -> pd.Series:
    """SMA crossover signal (-1, 0, 1)."""
    cross = np.zeros(len(short_sma))
    cross[short_sma > long_sma] = 1
    cross[short_sma < long_sma] = -1
    return pd.Series(cross, index=short_sma.index)


# ===========================================
# MOMENTUM INDICATORS
# ===========================================

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = safe_divide(avg_gain.values, avg_loss.values, default=100)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=close.index)


def calculate_rsi_category(rsi: pd.Series) -> pd.Series:
    """RSI category: -1 (oversold), 0 (neutral), 1 (overbought)."""
    category = np.zeros(len(rsi))
    category[rsi < 30] = -1  # Oversold
    category[rsi > 70] = 1   # Overbought
    return pd.Series(category, index=rsi.index)


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple:
    """MACD, Signal, Histogram."""
    ema_fast = calculate_ema(close, fast_period)
    ema_slow = calculate_ema(close, slow_period)
    macd = ema_fast - ema_slow
    signal = calculate_ema(macd, signal_period)
    histogram = macd - signal
    return macd, signal, histogram


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple:
    """Stochastic Oscillator (%K, %D)."""
    lowest_low = low.rolling(window=period, min_periods=1).min()
    highest_high = high.rolling(window=period, min_periods=1).max()
    
    stoch_k = safe_divide(
        (close - lowest_low).values,
        (highest_high - lowest_low).values,
        default=50
    ) * 100
    
    stoch_k = pd.Series(stoch_k, index=close.index).rolling(window=smooth_k, min_periods=1).mean()
    stoch_d = stoch_k.rolling(window=smooth_d, min_periods=1).mean()
    
    return stoch_k, stoch_d


def calculate_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp = calculate_sma(typical_price, period)
    mean_deviation = typical_price.rolling(window=period, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci = safe_divide(
        (typical_price - sma_tp).values,
        (0.015 * mean_deviation).values,
        default=0
    )
    return pd.Series(cci, index=close.index)


def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    tp_diff = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_diff > 0, 0).rolling(window=period, min_periods=1).sum()
    negative_flow = raw_money_flow.where(tp_diff < 0, 0).rolling(window=period, min_periods=1).sum()
    
    mfr = safe_divide(positive_flow.values, negative_flow.values, default=1)
    mfi = 100 - (100 / (1 + mfr))
    return pd.Series(mfi, index=close.index)


# ===========================================
# VOLATILITY INDICATORS
# ===========================================

def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: int = 2,
) -> tuple:
    """Bollinger Bands (upper, middle, lower, width, position)."""
    middle = calculate_sma(close, period)
    std = close.rolling(window=period, min_periods=1).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    width = safe_divide((upper - lower).values, middle.values, default=0) * 100
    
    # Position within bands (0-1)
    position = safe_divide(
        (close - lower).values,
        (upper - lower).values,
        default=0.5
    )
    
    return (
        upper,
        lower,
        pd.Series(width, index=close.index),
        pd.Series(position, index=close.index),
    )


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple:
    """Average True Range."""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    atr_pct = safe_divide(atr.values, close.values, default=0) * 100
    
    return atr, pd.Series(atr_pct, index=close.index)


# ===========================================
# TREND STRENGTH
# ===========================================

def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple:
    """ADX, +DI, -DI."""
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    # Smoothed
    tr_smooth = tr.rolling(window=period, min_periods=1).sum()
    plus_dm_smooth = plus_dm.rolling(window=period, min_periods=1).sum()
    minus_dm_smooth = minus_dm.rolling(window=period, min_periods=1).sum()
    
    # +DI, -DI
    plus_di = safe_divide(plus_dm_smooth.values, tr_smooth.values, default=0) * 100
    minus_di = safe_divide(minus_dm_smooth.values, tr_smooth.values, default=0) * 100
    
    # DX and ADX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = safe_divide(di_diff, di_sum, default=0) * 100
    
    adx = pd.Series(dx, index=close.index).rolling(window=period, min_periods=1).mean()
    
    return adx, pd.Series(plus_di, index=close.index), pd.Series(minus_di, index=close.index)


# ===========================================
# MAIN FEATURE COMPUTATION
# ===========================================

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators from OHLCV data.
    
    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
    
    Returns:
        DataFrame with all features added
    """
    logger.info("🔧 Computing technical indicators...")
    
    df = df.copy()
    periods = config.INDICATOR_PERIODS
    
    # Basic returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    
    # Trend indicators
    df["sma_10"] = calculate_sma(df["close"], periods["sma_short"])
    df["sma_20"] = calculate_sma(df["close"], periods["sma_medium"])
    df["sma_50"] = calculate_sma(df["close"], periods["sma_long"])
    df["ema_12"] = calculate_ema(df["close"], periods["ema_short"])
    df["ema_26"] = calculate_ema(df["close"], periods["ema_long"])
    
    df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"] * 100
    df["sma_cross"] = calculate_sma_cross(df["sma_10"], df["sma_50"])
    
    # Momentum indicators
    df["rsi"] = calculate_rsi(df["close"], periods["rsi"])
    df["rsi_category"] = calculate_rsi_category(df["rsi"])
    
    df["macd"], df["macd_signal"], df["macd_histogram"] = calculate_macd(
        df["close"],
        periods["macd_fast"],
        periods["macd_slow"],
        periods["macd_signal"],
    )
    
    df["stoch_k"], df["stoch_d"] = calculate_stochastic(
        df["high"], df["low"], df["close"], periods["stoch"]
    )
    
    df["cci"] = calculate_cci(df["high"], df["low"], df["close"], periods["cci"])
    df["mfi"] = calculate_mfi(df["high"], df["low"], df["close"], df["volume"], periods["mfi"])
    
    # Volatility indicators
    df["bb_upper"], df["bb_lower"], df["bb_width"], df["bb_position"] = calculate_bollinger_bands(
        df["close"], periods["bb_period"], periods["bb_std"]
    )
    df["atr"], df["atr_pct"] = calculate_atr(df["high"], df["low"], df["close"], periods["atr"])
    
    # Trend strength
    df["adx"], df["plus_di"], df["minus_di"] = calculate_adx(
        df["high"], df["low"], df["close"], periods["adx"]
    )
    
    # Volume indicators
    df["volume_sma"] = calculate_sma(df["volume"], 20)
    df["volume_ratio"] = df["volume"] / df["volume_sma"]
    
    # Time features
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    
    # Fill NaN values
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    logger.info(f"✅ Computed {len(config.FEATURE_COLUMNS)} features")
    return df


def create_target(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    """
    Create target variable: 1 if price goes UP, 0 if DOWN.
    
    Args:
        df: DataFrame with 'close' column
        horizon: Number of candles to look ahead
    
    Returns:
        Binary target series
    """
    future_close = df["close"].shift(-horizon)
    target = (future_close > df["close"]).astype(int)
    return target


if __name__ == "__main__":
    # Test feature engineering
    import data_fetcher
    
    df = data_fetcher.fetch_and_save_data()
    df = compute_all_features(df)
    
    print(f"\nFeatures computed: {df.columns.tolist()}")
    print(f"\nShape: {df.shape}")
    print(f"\nSample:\n{df[config.FEATURE_COLUMNS].head()}")
