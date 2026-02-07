"""
Trading Bot - Automated Trading Simulation

Consumes market data from Kafka, gets predictions from Model Server,
executes simulated trades, and stores results in TimescaleDB.
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, List

import asyncpg
import httpx
from aiokafka import AIOKafkaConsumer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'market_ticks')
MODEL_SERVER_URL = os.getenv('MODEL_SERVER_URL', 'http://model-server:8001')
TIMESCALE_HOST = os.getenv('TIMESCALE_HOST', 'timescaledb')
TIMESCALE_PORT = int(os.getenv('TIMESCALE_PORT', '5432'))
TIMESCALE_USER = os.getenv('TIMESCALE_USER', 'hft_user')
TIMESCALE_PASSWORD = os.getenv('TIMESCALE_PASSWORD', 'hft_password')
TIMESCALE_DB = os.getenv('TIMESCALE_DB', 'hft_trading')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.55'))
SYMBOL = os.getenv('SYMBOL', 'BTC/USDT')
TRADE_AMOUNT = float(os.getenv('TRADE_AMOUNT', '0.01'))  # BTC


class Position(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Represents a single trade."""
    time: datetime
    symbol: str
    side: str  # BUY or SELL
    price: float
    quantity: float
    confidence: float
    model_version: str
    latency_ms: float
    pnl: Optional[float] = None


@dataclass
class TradingStats:
    """Tracks trading performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    total_latency: float = 0.0
    predictions_count: int = 0
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100
    
    @property
    def avg_latency(self) -> float:
        if self.predictions_count == 0:
            return 0.0
        return self.total_latency / self.predictions_count
    
    def to_dict(self) -> dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 2),
            'total_pnl': round(self.total_pnl, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'avg_latency_ms': round(self.avg_latency, 2),
            'predictions_count': self.predictions_count,
        }


# FastAPI for stats endpoint
app = FastAPI(title="HFT Trading Bot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
trading_bot: Optional['TradingBot'] = None


class TradingBot:
    """Main trading bot logic."""
    
    def __init__(self):
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.running = True
        
        # Trading state
        self.position = Position.FLAT
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        
        # Stats
        self.stats = TradingStats()
        self.recent_trades: List[Trade] = []
        
    async def start(self) -> None:
        """Initialize all connections."""
        # HTTP client for model server
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # Connect to TimescaleDB
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
                    max_size=10,
                )
                logger.info(f"Connected to TimescaleDB at {TIMESCALE_HOST}")
                break
            except Exception as e:
                logger.warning(f"TimescaleDB connection attempt {attempt + 1}/{max_retries}: {e}")
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
                    group_id='trading-bot-group',
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='latest',
                )
                await self.consumer.start()
                logger.info(f"Connected to Kafka, consuming from: {KAFKA_TOPIC}")
                break
            except Exception as e:
                logger.warning(f"Kafka connection attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
                else:
                    raise
        
        # Wait for model server
        logger.info(f"Waiting for Model Server at {MODEL_SERVER_URL}...")
        for attempt in range(max_retries):
            try:
                response = await self.http_client.get(f"{MODEL_SERVER_URL}/health")
                if response.status_code == 200:
                    health = response.json()
                    if health.get('model_loaded'):
                        logger.info(f"Model Server ready, model version: {health.get('model_version')}")
                        break
            except Exception as e:
                pass
            if attempt < max_retries - 1:
                await asyncio.sleep(3)
    
    async def stop(self) -> None:
        """Gracefully stop."""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
        if self.db_pool:
            await self.db_pool.close()
        if self.http_client:
            await self.http_client.aclose()
        logger.info("Trading Bot stopped")
    
    async def get_prediction(self) -> Optional[dict]:
        """Get prediction from Model Server."""
        try:
            response = await self.http_client.get(f"{MODEL_SERVER_URL}/predict/live")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Prediction error: {e}")
        return None
    
    async def store_trade(self, trade: Trade) -> None:
        """Store trade in TimescaleDB."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO orders (time, symbol, side, price, quantity, status, pnl, model_version, confidence)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''', trade.time, trade.symbol, trade.side, trade.price, 
                    trade.quantity, 'EXECUTED', trade.pnl, trade.model_version, trade.confidence)
        except Exception as e:
            logger.error(f"Failed to store trade: {e}")
    
    def calculate_pnl(self, entry_price: float, exit_price: float, side: str) -> float:
        """Calculate PnL for a trade."""
        if side == "SELL":  # Closing LONG
            return (exit_price - entry_price) * TRADE_AMOUNT
        else:  # Closing SHORT
            return (entry_price - exit_price) * TRADE_AMOUNT
    
    def update_stats(self, pnl: float, latency: float) -> None:
        """Update trading statistics."""
        self.stats.total_trades += 1
        self.stats.total_pnl += pnl
        
        if pnl > 0:
            self.stats.winning_trades += 1
        else:
            self.stats.losing_trades += 1
        
        self.stats.max_pnl = max(self.stats.max_pnl, self.stats.total_pnl)
        self.stats.min_pnl = min(self.stats.min_pnl, self.stats.total_pnl)
        
        # Update peak and drawdown
        if self.stats.total_pnl > self.stats.peak_pnl:
            self.stats.peak_pnl = self.stats.total_pnl
        
        drawdown = self.stats.peak_pnl - self.stats.total_pnl
        self.stats.max_drawdown = max(self.stats.max_drawdown, drawdown)
        
        self.stats.total_latency += latency
        self.stats.predictions_count += 1
    
    async def execute_trade(self, prediction: dict, current_price: float) -> Optional[Trade]:
        """Execute trade based on prediction."""
        direction = prediction['prediction']  # UP or DOWN
        confidence = prediction['confidence']
        model_version = prediction['model_version']
        latency = prediction['latency_ms']
        
        self.stats.predictions_count += 1
        self.stats.total_latency += latency
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return None
        
        trade = None
        now = datetime.now(timezone.utc)
        
        # Trading logic
        if direction == "UP" and self.position != Position.LONG:
            # Close SHORT if open
            if self.position == Position.SHORT and self.entry_price:
                pnl = self.calculate_pnl(self.entry_price, current_price, "BUY")
                trade = Trade(
                    time=now, symbol=SYMBOL, side="BUY", price=current_price,
                    quantity=TRADE_AMOUNT, confidence=confidence,
                    model_version=model_version, latency_ms=latency, pnl=pnl
                )
                self.update_stats(pnl, latency)
            
            # Open LONG
            self.position = Position.LONG
            self.entry_price = current_price
            self.entry_time = now
            
            if trade is None:
                trade = Trade(
                    time=now, symbol=SYMBOL, side="BUY", price=current_price,
                    quantity=TRADE_AMOUNT, confidence=confidence,
                    model_version=model_version, latency_ms=latency
                )
                
        elif direction == "DOWN" and self.position != Position.SHORT:
            # Close LONG if open
            if self.position == Position.LONG and self.entry_price:
                pnl = self.calculate_pnl(self.entry_price, current_price, "SELL")
                trade = Trade(
                    time=now, symbol=SYMBOL, side="SELL", price=current_price,
                    quantity=TRADE_AMOUNT, confidence=confidence,
                    model_version=model_version, latency_ms=latency, pnl=pnl
                )
                self.update_stats(pnl, latency)
            
            # Open SHORT
            self.position = Position.SHORT
            self.entry_price = current_price
            self.entry_time = now
            
            if trade is None:
                trade = Trade(
                    time=now, symbol=SYMBOL, side="SELL", price=current_price,
                    quantity=TRADE_AMOUNT, confidence=confidence,
                    model_version=model_version, latency_ms=latency
                )
        
        return trade
    
    async def run(self) -> None:
        """Main trading loop."""
        tick_count = 0
        
        try:
            async for message in self.consumer:
                if not self.running:
                    break
                
                tick = message.value
                current_price = tick.get('price', 0)
                tick_count += 1
                
                # Get prediction every 5 ticks to reduce load
                if tick_count % 5 != 0:
                    continue
                
                prediction = await self.get_prediction()
                if prediction is None:
                    continue
                
                # Execute trade
                trade = await self.execute_trade(prediction, current_price)
                
                if trade:
                    await self.store_trade(trade)
                    self.recent_trades.append(trade)
                    if len(self.recent_trades) > 100:
                        self.recent_trades.pop(0)
                    
                    pnl_str = f"PnL: ${trade.pnl:.4f}" if trade.pnl else "OPEN"
                    logger.info(
                        f"{trade.side} {trade.quantity} {SYMBOL} @ ${trade.price:,.2f} | "
                        f"Conf: {trade.confidence:.1%} | {pnl_str} | "
                        f"Total: ${self.stats.total_pnl:.4f}"
                    )
                
                # Log stats every 50 predictions
                if self.stats.predictions_count % 50 == 0 and self.stats.predictions_count > 0:
                    logger.info(
                        f"Stats | Trades: {self.stats.total_trades} | "
                        f"Win Rate: {self.stats.win_rate:.1f}% | "
                        f"PnL: ${self.stats.total_pnl:.4f} | "
                        f"Drawdown: ${self.stats.max_drawdown:.4f}"
                    )
                    
        except Exception as e:
            if self.running:
                logger.error(f"Trading loop error: {e}")
                raise


# FastAPI endpoints
@app.get("/health")
async def health():
    return {"status": "running", "position": trading_bot.position.value if trading_bot else "unknown"}


@app.get("/stats")
async def get_stats():
    if trading_bot is None:
        return {"error": "Bot not initialized"}
    return trading_bot.stats.to_dict()


@app.get("/trades")
async def get_trades(limit: int = 20):
    if trading_bot is None:
        return {"error": "Bot not initialized"}
    trades = trading_bot.recent_trades[-limit:]
    return [
        {
            "time": t.time.isoformat(),
            "side": t.side,
            "price": t.price,
            "confidence": t.confidence,
            "pnl": t.pnl,
        }
        for t in trades
    ]


@app.get("/position")
async def get_position():
    if trading_bot is None:
        return {"error": "Bot not initialized"}
    return {
        "position": trading_bot.position.value,
        "entry_price": trading_bot.entry_price,
        "entry_time": trading_bot.entry_time.isoformat() if trading_bot.entry_time else None,
    }


async def run_bot():
    """Run the trading bot."""
    global trading_bot
    trading_bot = TradingBot()
    
    try:
        await trading_bot.start()
        logger.info("Trading Bot started!")
        await trading_bot.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
    finally:
        await trading_bot.stop()
        logger.info(f"Final Stats: {trading_bot.stats.to_dict()}")


def start_api():
    """Start FastAPI in background."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")


async def main():
    """Main entry point - run both API and bot."""
    import threading
    
    # Start API in background thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    logger.info("Stats API running on port 8002")
    
    # Run bot
    await run_bot()


if __name__ == "__main__":
    asyncio.run(main())
