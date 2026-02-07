"""
Trading Bot API Client

Fetches trading stats, trades, and position from Trading Bot service.
"""
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TradingStats:
    """Trading performance statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    avg_latency_ms: float = 0.0
    predictions_count: int = 0
    
    @property
    def profit_factor(self) -> float:
        """Gross profit / Gross loss."""
        if self.losing_trades == 0:
            return float('inf') if self.winning_trades > 0 else 0.0
        # Simplified: assume avg win = avg loss
        return self.winning_trades / max(self.losing_trades, 1)
    
    @property
    def avg_trade_pnl(self) -> float:
        """Average PnL per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades


@dataclass
class Trade:
    """Single trade record."""
    time: str
    side: str
    price: float
    confidence: float
    pnl: Optional[float] = None


@dataclass
class Position:
    """Current trading position."""
    position: str  # FLAT, LONG, SHORT
    entry_price: Optional[float] = None
    entry_time: Optional[str] = None


class TradingClient:
    """Client for Trading Bot API."""
    
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def get_stats(self) -> Optional[TradingStats]:
        """Fetch trading statistics."""
        try:
            response = self._client.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                data = response.json()
                return TradingStats(
                    total_trades=data.get('total_trades', 0),
                    winning_trades=data.get('winning_trades', 0),
                    losing_trades=data.get('losing_trades', 0),
                    win_rate=data.get('win_rate', 0.0),
                    total_pnl=data.get('total_pnl', 0.0),
                    max_drawdown=data.get('max_drawdown', 0.0),
                    avg_latency_ms=data.get('avg_latency_ms', 0.0),
                    predictions_count=data.get('predictions_count', 0),
                )
        except Exception as e:
            logger.error(f"Failed to fetch stats: {e}")
        return None
    
    def get_trades(self, limit: int = 20) -> List[Trade]:
        """Fetch recent trades."""
        try:
            response = self._client.get(f"{self.base_url}/trades", params={"limit": limit})
            if response.status_code == 200:
                data = response.json()
                # Handle both list and dict with 'value' key
                trades_data = data if isinstance(data, list) else data.get('value', [])
                return [
                    Trade(
                        time=t.get('time', ''),
                        side=t.get('side', ''),
                        price=t.get('price', 0),
                        confidence=t.get('confidence', 0),
                        pnl=t.get('pnl'),
                    )
                    for t in trades_data
                ]
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
        return []
    
    def get_position(self) -> Optional[Position]:
        """Fetch current position."""
        try:
            response = self._client.get(f"{self.base_url}/position")
            if response.status_code == 200:
                data = response.json()
                return Position(
                    position=data.get('position', 'FLAT'),
                    entry_price=data.get('entry_price'),
                    entry_time=data.get('entry_time'),
                )
        except Exception as e:
            logger.error(f"Failed to fetch position: {e}")
        return None
    
    def is_healthy(self) -> bool:
        """Check if trading bot is healthy."""
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
