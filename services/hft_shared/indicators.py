"""
Technical Indicators Library (Shared)

Centralized logic for calculating technical indicators to ensure consistency
between Feature Calculator (Serving) and Data Generator (Training).

Optimized for performance using numpy.
"""
import numpy as np
from typing import Tuple

class TechnicalIndicators:
    """
    CPU-optimized technical indicator calculations.
    Uses pure numpy for speed without heavy dependencies.
    """
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> float:
        """Simple Moving Average."""
        if len(prices) < period:
            return float(np.mean(prices)) if len(prices) > 0 else 0.0
        return float(np.mean(prices[-period:]))
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> float:
        """Exponential Moving Average (optimized)."""
        if len(prices) < 2:
            return prices[-1] if len(prices) > 0 else 0.0
        
        alpha = 2.0 / (period + 1)
        ema_val = prices[0]
        for price in prices[1:]:
            ema_val = alpha * price + (1 - alpha) * ema_val
        return float(ema_val)
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))
    
    @staticmethod
    def macd(prices: np.ndarray) -> Tuple[float, float, float]:
        """MACD indicator (12, 26, 9)."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        ema_12 = TechnicalIndicators.ema(prices, 12)
        ema_26 = TechnicalIndicators.ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        # Signal line (EMA of MACD) - simplified approximation for streaming
        signal = macd_line * 0.2  
        histogram = macd_line - signal
        
        return float(macd_line), float(signal), float(histogram)
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float, float]:
        """Bollinger Bands."""
        if len(prices) < period:
            price = prices[-1] if len(prices) > 0 else 0.0
            return price, price, price, 0.0
        
        sma = float(np.mean(prices[-period:]))
        std = float(np.std(prices[-period:]))
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = ((upper - lower) / sma * 100) if sma > 0 else 0.0
        
        return upper, sma, lower, width

    @staticmethod
    def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Average True Range."""
        if len(highs) < 2:
            return 0.0
        
        tr_list = []
        # Calculate TR for the window needed
        window_size = min(len(highs), period + 1)
        
        for i in range(1, window_size):
            # Index from end: -1 is current, -2 is previous
            idx = -i 
            prev_idx = -(i + 1)
            
            if abs(prev_idx) > len(closes):
                 break

            high = highs[idx]
            low = lows[idx]
            prev_close = closes[prev_idx]
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        
        return float(np.mean(tr_list)) if tr_list else 0.0

    @staticmethod
    def roc(prices: np.ndarray, period: int = 10) -> float:
        """Rate of Change."""
        if len(prices) <= period:
            return 0.0
        
        current = prices[-1]
        past = prices[-period - 1]
        
        if past == 0:
            return 0.0
        
        return float(((current - past) / past) * 100)
    
    @staticmethod
    def obv(prices: np.ndarray, volumes: np.ndarray) -> float:
        """On-Balance Volume (simplified windowed version)."""
        if len(prices) < 2:
            return 0.0
        
        obv_val = 0.0
        # Iterate through the prices window
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv_val += volumes[i]
            elif prices[i] < prices[i - 1]:
                obv_val -= volumes[i]
        
        return float(obv_val)
