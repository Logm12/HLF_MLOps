"""
Kafka Consumer for Testing
Consumes market tick data from Kafka and logs it to verify data flow.

This consumer is used to test that the Market Producer is correctly
publishing data to the Kafka topic.
"""
import asyncio
import json
import logging
import os
import signal
from datetime import datetime
from typing import Optional

from aiokafka import AIOKafkaConsumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'market_ticks')
CONSUMER_GROUP = os.getenv('CONSUMER_GROUP', 'test-consumer-group')


class MarketConsumer:
    """
    Consumes market tick data from Kafka for testing purposes.
    
    Features:
    - Automatic reconnection on failure
    - Graceful shutdown handling
    - Message counting and latency tracking
    """
    
    def __init__(self):
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.running = True
        self.message_count = 0
        self.total_latency_ms = 0.0
        
    async def start_consumer(self) -> None:
        """Initialize Kafka consumer with retry logic."""
        max_retries = 15
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.consumer = AIOKafkaConsumer(
                    KAFKA_TOPIC,
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    group_id=CONSUMER_GROUP,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    auto_commit_interval_ms=1000,
                )
                await self.consumer.start()
                logger.info(f"✅ Connected to Kafka, listening on topic: {KAFKA_TOPIC}")
                return
            except Exception as e:
                logger.warning(
                    f"⚠️ Connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("❌ Failed to connect to Kafka after all retries")
                    raise
    
    async def stop_consumer(self) -> None:
        """Gracefully stop the consumer."""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
            logger.info("🛑 Consumer stopped")
    
    def calculate_latency(self, tick: dict) -> float:
        """Calculate end-to-end latency in milliseconds."""
        try:
            tick_time = datetime.fromisoformat(tick['timestamp'].replace('Z', '+00:00'))
            now = datetime.now(tick_time.tzinfo)
            latency = (now - tick_time).total_seconds() * 1000
            return max(0, latency)
        except Exception:
            return 0.0
    
    async def consume(self) -> None:
        """Consume messages from Kafka topic."""
        try:
            async for message in self.consumer:
                if not self.running:
                    break
                    
                tick = message.value
                self.message_count += 1
                
                # Calculate latency
                latency_ms = self.calculate_latency(tick)
                self.total_latency_ms += latency_ms
                avg_latency = self.total_latency_ms / self.message_count
                
                # Log message details
                logger.info(
                    f"📨 [{self.message_count:>5}] {tick['symbol']} | "
                    f"Price: ${tick['price']:>10,.2f} | "
                    f"Bid: ${tick['bid']:>10,.2f} | "
                    f"Ask: ${tick['ask']:>10,.2f} | "
                    f"Change: {tick['price_change_pct']:>+6.2f}% | "
                    f"Latency: {latency_ms:>6.1f}ms (avg: {avg_latency:.1f}ms)"
                )
                
        except Exception as e:
            if self.running:
                logger.error(f"❌ Consumer error: {e}")
                raise


async def main():
    """Main entry point with graceful shutdown handling."""
    consumer = MarketConsumer()
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    
    def shutdown_handler():
        logger.info("🛑 Shutdown signal received...")
        consumer.running = False
    
    try:
        loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
        loop.add_signal_handler(signal.SIGINT, shutdown_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass
    
    try:
        await consumer.start_consumer()
        logger.info(f"🚀 Starting to consume from {KAFKA_TOPIC}...")
        await consumer.consume()
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        await consumer.stop_consumer()
        
        # Print statistics
        if consumer.message_count > 0:
            avg_latency = consumer.total_latency_ms / consumer.message_count
            logger.info(f"📊 Statistics:")
            logger.info(f"   Total messages: {consumer.message_count}")
            logger.info(f"   Average latency: {avg_latency:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
