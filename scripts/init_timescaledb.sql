-- ============================================
-- TimescaleDB Initialization Script
-- HFT MLOps System
-- ============================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================
-- MARKET TICKS TABLE
-- Raw price data from Binance
-- ============================================
CREATE TABLE IF NOT EXISTS market_ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION,
    high_24h DOUBLE PRECISION,
    low_24h DOUBLE PRECISION,
    price_change_24h DOUBLE PRECISION,
    price_change_pct DOUBLE PRECISION,
    source VARCHAR(50) DEFAULT 'binance'
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_ticks', 'time', if_not_exists => TRUE);

-- Index for fast symbol lookups
CREATE INDEX IF NOT EXISTS idx_market_ticks_symbol ON market_ticks (symbol, time DESC);

-- ============================================
-- ORDERS TABLE
-- Trading orders and execution history (Phase 4)
-- ============================================
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'BUY' or 'SELL'
    price DOUBLE PRECISION NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    status VARCHAR(20) DEFAULT 'EXECUTED',
    pnl DOUBLE PRECISION,
    model_version VARCHAR(100),
    confidence DOUBLE PRECISION,
    strategy VARCHAR(50)
);

SELECT create_hypertable('orders', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol, time DESC);

-- ============================================
-- FEATURE SNAPSHOTS TABLE
-- For debugging and analysis
-- ============================================
CREATE TABLE IF NOT EXISTS feature_snapshots (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    features JSONB NOT NULL
);

SELECT create_hypertable('feature_snapshots', 'time', if_not_exists => TRUE);

-- ============================================
-- OHLCV CANDLES TABLE
-- Aggregated candlestick data (Phase 2)
-- ============================================
CREATE TABLE IF NOT EXISTS ohlcv_1m (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('ohlcv_1m', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_1m (symbol, time DESC);

-- ============================================
-- PERFORMANCE METRICS TABLE
-- System latency and performance tracking (Phase 5)
-- ============================================
CREATE TABLE IF NOT EXISTS performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    tags JSONB
);

SELECT create_hypertable('performance_metrics', 'time', if_not_exists => TRUE);

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB initialization completed successfully!';
END $$;
