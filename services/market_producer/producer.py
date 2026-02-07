"""
Market Data Producer
Fetches real-time price data from Binance WebSocket and publishes to Kafka.

This producer connects to Binance's public WebSocket API and streams
BTC/USDT ticker data to a Kafka topic in real-time.
"""
import asyncio
import json
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Optional

import websockets
from aiokafka import AIOKafkaProducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'market_ticks')
SYMBOL = os.getenv('SYMBOL', 'BTC/USDT')

# Binance WebSocket URL for ticker stream
# Using 24hr ticker stream which updates every second
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@ticker"


class MarketProducer:
    """
    Produces market tick data from Binance WebSocket to Kafka.
    
    Features:
    - Automatic reconnection on WebSocket failure
    - Graceful shutdown handling
    - Message counting and logging
    - Exactly-once semantics with idempotent producer
    """
    
    def __init__(self):
        self.producer: Optional[AIOKafkaProducer] = None
        self.running = True
        self.message_count = 0
        self.last_price = 0.0
        
    async def start_producer(self) -> None:
        """Initialize Kafka producer with retry logic."""
        max_retries = 15
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.producer = AIOKafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    enable_idempotence=True,
                    max_batch_size=16384,
                    linger_ms=10,
                )
                await self.producer.start()
                logger.info(f"✅ Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
                return
            except Exception as e:
                logger.warning(
                    f"⚠️ Kafka connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("❌ Failed to connect to Kafka after all retries")
                    raise
    
    async def stop_producer(self) -> None:
        """Gracefully stop the producer."""
        self.running = False
        if self.producer:
            await self.producer.stop()
            logger.info("🛑 Kafka producer stopped")
    
    def parse_binance_ticker(self, data: dict) -> dict:
        """
        Parse Binance ticker data into standardized format.
        
        Binance ticker fields:
        - c: Close price (current price)
        - v: Total traded base asset volume
        - b: Best bid price
        - a: Best ask price
        - h: High price
        - l: Low price
        - p: Price change
        - P: Price change percent
        """
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_time': data.get('E'),
            'symbol': SYMBOL,
            'price': float(data.get('c', 0)),
            'volume': float(data.get('v', 0)),
            'bid': float(data.get('b', 0)),
            'ask': float(data.get('a', 0)),
            'high_24h': float(data.get('h', 0)),
            'low_24h': float(data.get('l', 0)),
            'price_change_24h': float(data.get('p', 0)),
            'price_change_pct': float(data.get('P', 0)),
            'trades_count': int(data.get('n', 0)),
            'source': 'binance'
        }
    
    async def stream_market_data(self) -> None:
        """Connect to Binance WebSocket and stream data to Kafka."""
        while self.running:
            try:
                logger.info(f"🔌 Connecting to Binance WebSocket...")
                async with websockets.connect(
                    BINANCE_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    logger.info(f"✅ Connected to Binance WebSocket for {SYMBOL}")
                    
                    while self.running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(message)
                            
                            # Parse and publish to Kafka
                            tick = self.parse_binance_ticker(data)
                            self.last_price = tick['price']
                            
                            await self.producer.send_and_wait(
                                topic=KAFKA_TOPIC,
                                key=SYMBOL,
                                value=tick
                            )
                            
                            self.message_count += 1
                            
                            # Log every 100 messages
                            if self.message_count % 100 == 0:
                                logger.info(
                                    f"📊 [{self.message_count}] {SYMBOL} | "
                                    f"Price: ${tick['price']:,.2f} | "
                                    f"Change: {tick['price_change_pct']:+.2f}%"
                                )
                                
                        except asyncio.TimeoutError:
                            logger.warning("⚠️ WebSocket timeout, sending ping...")
                            try:
                                pong = await ws.ping()
                                await asyncio.wait_for(pong, timeout=10)
                            except Exception:
                                logger.warning("⚠️ Ping failed, reconnecting...")
                                break
                            
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"⚠️ WebSocket connection closed: {e}")
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
            
            if self.running:
                logger.info("🔄 Reconnecting in 5 seconds...")
                await asyncio.sleep(5)


async def main():
    """Main entry point with graceful shutdown handling."""
    producer = MarketProducer()
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    
    def shutdown_handler():
        logger.info("🛑 Shutdown signal received...")
        producer.running = False
    
    # Handle SIGTERM for Docker
    try:
        loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
        loop.add_signal_handler(signal.SIGINT, shutdown_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass
    
    try:
        await producer.start_producer()
        logger.info(f"🚀 Starting market data stream for {SYMBOL}...")
        await producer.stream_market_data()
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        await producer.stop_producer()
        logger.info(f"📈 Total messages published: {producer.message_count}")


if __name__ == "__main__":
    asyncio.run(main())
