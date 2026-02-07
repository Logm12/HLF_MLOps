"""
Feature Calculator Service

Consumes market tick data from Kafka, calculates technical indicators,
and pushes features to Redis via Feast for low-latency model access.

CPU-optimized: Uses lightweight numpy calculations instead of heavy libraries.
"""
import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

import numpy as np
import redis
from aiokafka import AIOKafkaConsumer
from prometheus_client import start_http_server, Counter, Histogram

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'market_ticks')
CONSUMER_GROUP = os.getenv('CONSUMER_GROUP', 'feature-calculator-group')
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))

# Buffer size for indicator calculations (lightweight for CPU)
BUFFER_SIZE = 50  # Enough for most indicators

# Metrics
MESSAGES_PROCESSED = Counter('messages_processed_total', 'Total market ticks processed')
CALCULATION_LATENCY = Histogram('calculation_latency_seconds', 'Time spent calculating features')
BUFFER_SIZE_METRIC = Histogram('buffer_size', 'Current buffer size')


@dataclass
class PriceBuffer:
    """Efficient price buffer using deque for O(1) append/pop."""
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    volumes: Deque[float] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    highs: Deque[float] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    lows: Deque[float] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))


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
    def macd(prices: np.ndarray) -> tuple:
        """MACD indicator (12, 26, 9)."""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        ema_12 = TechnicalIndicators.ema(prices, 12)
        ema_26 = TechnicalIndicators.ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        # Signal line (EMA of MACD) - simplified
        signal = macd_line * 0.2  # Approximation for speed
        histogram = macd_line - signal
        
        return float(macd_line), float(signal), float(histogram)
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
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
        for i in range(1, min(len(highs), period + 1)):
            high = highs[-i]
            low = lows[-i]
            prev_close = closes[-(i + 1)] if i + 1 <= len(closes) else closes[-i]
            
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
        """On-Balance Volume."""
        if len(prices) < 2:
            return 0.0
        
        obv_val = 0.0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv_val += volumes[i]
            elif prices[i] < prices[i - 1]:
                obv_val -= volumes[i]
        
        return float(obv_val)


class FeatureCalculator:
    """
    Main service that consumes Kafka data and produces features.
    Optimized for low-latency CPU-based calculation.
    """
    
    def __init__(self):
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.redis_client: Optional[redis.Redis] = None
        self.running = True
        self.buffers: Dict[str, PriceBuffer] = {}
        self.message_count = 0
        self.total_latency = 0.0
        self.indicators = TechnicalIndicators()
        
    async def start(self) -> None:
        """Initialize Kafka consumer and Redis client."""
        # Connect to Redis
        max_retries = 15
        for attempt in range(max_retries):
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                self.redis_client.ping()
                logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Redis connection attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                else:
                    raise
        
        # Connect to Kafka
        for attempt in range(max_retries):
            try:
                self.consumer = AIOKafkaConsumer(
                    KAFKA_TOPIC,
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    group_id=CONSUMER_GROUP,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    max_poll_interval_ms=300000,
                )
                await self.consumer.start()
                logger.info(f"✅ Connected to Kafka, consuming from: {KAFKA_TOPIC}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Kafka connection attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                else:
                    raise
    
    async def stop(self) -> None:
        """Gracefully stop the service."""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
        if self.redis_client:
            self.redis_client.close()
        logger.info("🛑 Feature Calculator stopped")
    
    def get_or_create_buffer(self, symbol: str) -> PriceBuffer:
        """Get or create price buffer for symbol."""
        if symbol not in self.buffers:
            self.buffers[symbol] = PriceBuffer()
        return self.buffers[symbol]
    
    def calculate_features(self, symbol: str, tick: dict) -> dict:
        """Calculate all technical indicators from price buffer."""
        start_time = time.perf_counter()
        
        buffer = self.get_or_create_buffer(symbol)
        
        # Update buffer
        price = tick['price']
        volume = tick.get('volume', 0)
        high = tick.get('high_24h', price)
        low = tick.get('low_24h', price)
        
        buffer.prices.append(price)
        buffer.volumes.append(volume)
        buffer.highs.append(high)
        buffer.lows.append(low)
        buffer.timestamps.append(time.time())
        
        # Convert to numpy arrays
        prices = np.array(buffer.prices)
        volumes = np.array(buffer.volumes)
        highs = np.array(buffer.highs)
        lows = np.array(buffer.lows)
        
        # Calculate indicators
        sma_20 = self.indicators.sma(prices, 20)
        ema_12 = self.indicators.ema(prices, 12)
        ema_26 = self.indicators.ema(prices, 26)
        rsi_14 = self.indicators.rsi(prices, 14)
        macd, macd_signal, macd_histogram = self.indicators.macd(prices)
        bb_upper, bb_middle, bb_lower, bb_width = self.indicators.bollinger_bands(prices, 20)
        atr_14 = self.indicators.atr(highs, lows, prices, 14)
        roc_10 = self.indicators.roc(prices, 10)
        obv = self.indicators.obv(prices, volumes)
        volume_sma = self.indicators.sma(volumes, 20)
        
        # Derived features
        spread = tick.get('ask', price) - tick.get('bid', price)
        spread_pct = (spread / price * 100) if price > 0 else 0
        price_vs_sma = ((price - sma_20) / sma_20 * 100) if sma_20 > 0 else 0
        volume_ratio = (volume / volume_sma) if volume_sma > 0 else 1.0
        
        # Calculate latency
        calc_latency = (time.perf_counter() - start_time) * 1000
        
        # Data age (time since tick was generated)
        try:
            tick_time = datetime.fromisoformat(tick['timestamp'].replace('Z', '+00:00'))
            data_age = (datetime.now(timezone.utc) - tick_time).total_seconds() * 1000
        except Exception:
            data_age = 0
        
        features = {
            # Price features
            'price_current': price,
            'price_change_pct': tick.get('price_change_pct', 0),
            'volume': volume,
            'bid': tick.get('bid', price),
            'ask': tick.get('ask', price),
            'spread': spread,
            'spread_pct': spread_pct,
            
            # Trend indicators
            'sma_20': sma_20,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'price_vs_sma': price_vs_sma,
            
            # Momentum indicators
            'rsi_14': rsi_14,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'roc_10': roc_10,
            
            # Volatility indicators
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'atr_14': atr_14,
            
            # Volume indicators
            'obv': obv,
            'volume_sma': volume_sma,
            'volume_ratio': volume_ratio,
            
            # System metrics
            'latency_ms': calc_latency,
            'data_age_ms': data_age,
            'buffer_size': len(buffer.prices),
        }
        
        # Metrics
        MESSAGES_PROCESSED.inc()
        CALCULATION_LATENCY.observe(calc_latency / 1000.0)
        
        return features
    
    def push_to_redis(self, symbol: str, features: dict) -> None:
        """Push features to Redis for Feast online store."""
        try:
            # Store as hash for efficient retrieval
            key = f"hft:features:{symbol.replace('/', '_')}"
            
            # Convert all values to strings for Redis
            redis_data = {k: str(v) for k, v in features.items()}
            redis_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            self.redis_client.hset(key, mapping=redis_data)
            self.redis_client.expire(key, 3600)  # 1 hour TTL
            
            # Also store latest price for quick access
            self.redis_client.set(
                f"hft:price:{symbol.replace('/', '_')}",
                str(features['price_current']),
                ex=60
            )
            
        except Exception as e:
            logger.error(f"❌ Redis push error: {e}")
    
    async def run(self) -> None:
        """Main processing loop."""
        try:
            async for message in self.consumer:
                if not self.running:
                    break
                
                tick = message.value
                symbol = tick.get('symbol', 'BTC/USDT')
                
                # Calculate features
                features = self.calculate_features(symbol, tick)
                
                # Push to Redis
                self.push_to_redis(symbol, features)
                
                self.message_count += 1
                self.total_latency += features['latency_ms']
                
                # Log every 50 messages
                if self.message_count % 50 == 0:
                    avg_latency = self.total_latency / self.message_count
                    logger.info(
                        f"📊 [{self.message_count}] {symbol} | "
                        f"Price: ${features['price_current']:,.2f} | "
                        f"RSI: {features['rsi_14']:.1f} | "
                        f"MACD: {features['macd']:.2f} | "
                        f"Latency: {features['latency_ms']:.2f}ms (avg: {avg_latency:.2f}ms)"
                    )
                    
        except Exception as e:
            if self.running:
                logger.error(f"❌ Processing error: {e}")
                raise


async def main():
    """Main entry point."""
    # Start Prometheus metrics server
    start_http_server(8000)
    logger.info("📊 Metrics server started on port 8000")
    
    calculator = FeatureCalculator()
    
    try:
        await calculator.start()
        logger.info("🚀 Feature Calculator started, processing market data...")
        await calculator.run()
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        await calculator.stop()
        if calculator.message_count > 0:
            avg_latency = calculator.total_latency / calculator.message_count
            logger.info(f"📈 Processed {calculator.message_count} messages, avg latency: {avg_latency:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
