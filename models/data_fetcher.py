"""
Data Fetcher - Binance Historical OHLCV

Fetches real historical data from Binance public API.
Modular design for easy testing and reuse.
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from dataclasses import dataclass

import requests
import pandas as pd

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OHLCVCandle:
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BinanceDataFetcher:
    """
    Fetches historical OHLCV data from Binance.
    
    Uses public API (no API key required).
    Rate limit: 1200 requests/minute.
    """
    
    def __init__(self, base_url: str = config.BINANCE_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        multipliers = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        return multipliers.get(timeframe, 15 * 60 * 1000)
    
    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[OHLCVCandle]:
        """
        Fetch klines (candlestick data) from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candle interval ('1m', '5m', '15m', '1h', etc.)
            start_time: Start datetime (optional)
            end_time: End datetime (optional)
            limit: Max candles per request (max 1000)
        
        Returns:
            List of OHLCVCandle objects
        """
        url = f"{self.base_url}/api/v3/klines"
        
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": min(limit, 1000),
        }
        
        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            candles = []
            for kline in data:
                candle = OHLCVCandle(
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                )
                candles.append(candle)
            
            return candles
            
        except Exception as e:
            logger.error(f"Failed to fetch klines: {e}")
            return []
    
    def fetch_full_history(
        self,
        symbol: str,
        timeframe: str,
        days: int = 730,
    ) -> pd.DataFrame:
        """
        Fetch complete historical data.
        
        Args:
            symbol: Trading pair
            timeframe: Candle interval
            days: Number of days to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"📊 Fetching {days} days of {symbol} {timeframe} data...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        all_candles = []
        current_start = start_time
        batch_ms = self._timeframe_to_ms(timeframe) * 1000
        
        while current_start < end_time:
            candles = self.fetch_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_time=current_start,
                limit=1000,
            )
            
            if not candles:
                break
            
            all_candles.extend(candles)
            current_start = candles[-1].timestamp + timedelta(milliseconds=self._timeframe_to_ms(timeframe))
            
            # Progress logging
            progress = (current_start - start_time) / (end_time - start_time) * 100
            logger.info(f"   Progress: {progress:.1f}% ({len(all_candles)} candles)")
            
            # Rate limiting (be nice to Binance)
            time.sleep(0.2)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in all_candles
        ])
        
        if not df.empty:
            df = df.drop_duplicates(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"✅ Fetched {len(df)} candles from {start_time.date()} to {end_time.date()}")
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame to Parquet file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath, index=False)
        logger.info(f"💾 Saved to {filepath}")
    
    def load_from_parquet(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from Parquet file."""
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            logger.info(f"📂 Loaded {len(df)} rows from {filepath}")
            return df
        return None


def fetch_and_save_data() -> pd.DataFrame:
    """Main function to fetch and save data."""
    fetcher = BinanceDataFetcher()
    
    # Try to load existing data first
    existing_df = fetcher.load_from_parquet(config.DATA_FILE)
    if existing_df is not None and len(existing_df) > 50000:
        logger.info("Using cached data")
        return existing_df
    
    # Fetch new data
    df = fetcher.fetch_full_history(
        symbol=config.SYMBOL,
        timeframe=config.TIMEFRAME,
        days=config.LOOKBACK_DAYS,
    )
    
    if not df.empty:
        fetcher.save_to_parquet(df, config.DATA_FILE)
    
    return df


if __name__ == "__main__":
    # Test data fetching
    df = fetch_and_save_data()
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
