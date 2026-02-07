"""
Data Ingestor Service

Consumes market ticks from Kafka and persists them to TimescaleDB/PostgreSQL.
Ensures we have a real historical dataset for training.
"""
import asyncio
import json
import logging
import os
import signal
from typing import Optional

import asyncpg
from aiokafka import AIOKafkaConsumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'market_ticks')
GROUP_ID = os.getenv('CONSUMER_GROUP', 'data-ingestor-group')

TIMESCALE_HOST = os.getenv('TIMESCALE_HOST', 'timescaledb')
TIMESCALE_PORT = int(os.getenv('TIMESCALE_PORT', '5432'))
TIMESCALE_USER = os.getenv('TIMESCALE_USER', 'hft_user')
TIMESCALE_PASSWORD = os.getenv('TIMESCALE_PASSWORD', 'hft_password')
TIMESCALE_DB = os.getenv('TIMESCALE_DB', 'hft_trading')

class DataIngestor:
    def __init__(self):
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        self.running = True
        self.message_count = 0
        
    async def start(self):
        """Initialize connections to Kafka and DB."""
        # Connect to DB
        max_retries = 30
        for attempt in range(max_retries):
            try:
                self.db_pool = await asyncpg.create_pool(
                    host=TIMESCALE_HOST,
                    port=TIMESCALE_PORT,
                    user=TIMESCALE_USER,
                    password=TIMESCALE_PASSWORD,
                    database=TIMESCALE_DB,
                    min_size=2,
                    max_size=5
                )
                logger.info(f"✅ Connected to TimescaleDB at {TIMESCALE_HOST}")
                
                 # Initialize table if not exists (Idempotency)
                async with self.db_pool.acquire() as conn:
                    await conn.execute('''
                        CREATE TABLE IF NOT EXISTS market_ticks (
                            time TIMESTAMPTZ NOT NULL,
                            symbol TEXT NOT NULL,
                            price DOUBLE PRECISION NOT NULL,
                            volume DOUBLE PRECISION NOT NULL,
                            bid DOUBLE PRECISION,
                            ask DOUBLE PRECISION,
                            source TEXT
                        );
                        SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);
                    ''')
                break
            except Exception as e:
                logger.warning(f"⚠️ DB Connection attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise
                    
        # Connect to Kafka
        for attempt in range(max_retries):
            try:
                self.consumer = AIOKafkaConsumer(
                    KAFKA_TOPIC,
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    group_id=GROUP_ID,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                )
                await self.consumer.start()
                logger.info(f"✅ Connected to Kafka topic {KAFKA_TOPIC}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Kafka Connection attempt {attempt+1}/{max_retries}: {e}")
                await asyncio.sleep(2)

    async def stop(self):
        self.running = False
        if self.consumer:
            await self.consumer.stop()
        if self.db_pool:
            await self.db_pool.close()
        logger.info("🛑 Data Ingestor stopped")
        
    async def run(self):
        """Main ingestion loop."""
        batch = []
        BATCH_SIZE = 100
        
        try:
            async for message in self.consumer:
                if not self.running: 
                    break
                    
                tick = message.value
                # Normalize timestamp
                if 'timestamp' in tick:
                    ts = tick['timestamp'] # Expect ISO string
                else:
                    # Fallback
                    from datetime import datetime, timezone
                    ts = datetime.now(timezone.utc).isoformat()
                
                # Prepare row: (time, symbol, price, volume, bid, ask, source)
                row = (
                    ts,
                    tick.get('symbol'),
                    float(tick.get('price', 0)),
                    float(tick.get('volume', 0)),
                    float(tick.get('bid', 0)),
                    float(tick.get('ask', 0)),
                    tick.get('source', 'unknown')
                )
                batch.append(row)
                
                if len(batch) >= BATCH_SIZE:
                    await self.flush_batch(batch)
                    self.message_count += len(batch)
                    batch = []
                    if self.message_count % 1000 == 0:
                        logger.info(f"📥 Ingested {self.message_count} ticks total")
                        
            # Flush remaining
            if batch:
                await self.flush_batch(batch)
                
        except Exception as e:
            logger.error(f"❌ Ingestion loop handling error: {e}")
            raise

    async def flush_batch(self, batch):
        if not self.db_pool: return
        try:
            async with self.db_pool.acquire() as conn:
                # Type conversion might be needed depending on asyncpg strictness, 
                # strictly speaking isoformat string -> TIMESTAMPTZ works usually via casting
                await conn.executemany('''
                    INSERT INTO market_ticks (time, symbol, price, volume, bid, ask, source)
                    VALUES ($1::timestamptz, $2, $3, $4, $5, $6, $7)
                ''', batch)
        except Exception as e:
            logger.error(f"⚠️ Batch insert failed: {e}")

async def main():
    ingestor = DataIngestor()
    
    loop = asyncio.get_running_loop()
    stop_signal = asyncio.Event()
    
    def signal_handler():
        logger.info("🛑 Signal received")
        stop_signal.set()
        
    try:
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
        loop.add_signal_handler(signal.SIGINT, signal_handler)
    except NotImplementedError:
        pass
        
    try:
        await ingestor.start()
        logger.info("🚀 Data Ingestor started")
        
        ingest_task = asyncio.create_task(ingestor.run())
        
        # Wait for stop signal
        while not stop_signal.is_set():
            await asyncio.sleep(1)
            if ingest_task.done():
                break
                
        ingestor.running = False
        await ingest_task
        
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
    finally:
        await ingestor.stop()

if __name__ == "__main__":
    asyncio.run(main())
