"""
Model Server - FastAPI Real-time Inference Service

Loads the trained model from MLflow and serves predictions
using features from Redis (computed by Feature Calculator).
"""
import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np
import redis
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MODEL_NAME = os.getenv('MODEL_NAME', 'hft-xgboost-classifier')
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
SYMBOL = os.getenv('SYMBOL', 'BTC/USDT')

# Feature columns expected by model (must match training)
FEATURE_COLUMNS = [
    'rsi_14',
    'macd',
    'macd_signal',
    'macd_histogram',
    'bb_width',
    'price_vs_sma',
    'roc_10',
    'volume_ratio',
    'spread_pct',
    'ema_12',
    'ema_26',
]

# Initialize FastAPI app
app = FastAPI(
    title="HFT Model Server",
    description="Real-time price prediction API for High-Frequency Trading",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_version = None
redis_client = None
prediction_count = 0
correct_predictions = 0
total_latency = 0.0


# Metrics
PREDICTION_COUNT = Counter('prediction_count', 'Total number of predictions')
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
CONFIDENCE = Histogram('prediction_confidence', 'Prediction confidence level')

# Request/Response models
class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    symbol: str
    prediction: str
    confidence: float
    direction: int  # 1 = UP, 0 = DOWN
    features: Dict[str, float]
    latency_ms: float
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    redis_connected: bool
    predictions_count: int


class MetricsResponse(BaseModel):
    predictions_count: int
    average_latency_ms: float
    model_version: str
    uptime_seconds: float


# Startup time for uptime calculation
startup_time = None


@app.on_event("startup")
async def startup_event():
    """Load model and connect to Redis on startup."""
    global model, model_version, redis_client, startup_time
    
    startup_time = time.time()
    logger.info("Starting Model Server...")
    
    # Connect to Redis
    max_retries = 30
    for attempt in range(max_retries):
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_timeout=5,
            )
            redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            break
        except Exception as e:
            logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logger.error("Failed to connect to Redis")
    
    # Load model from MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    for attempt in range(max_retries):
        try:
            # Get latest model version
            client = mlflow.tracking.MlflowClient()
            
            # Try to get latest version
            try:
                versions = client.get_latest_versions(MODEL_NAME)
                if versions:
                    latest_version = versions[0]
                    model_version = latest_version.version
                    model_uri = f"models:/{MODEL_NAME}/{model_version}"
                else:
                    # Fallback: use model from latest run
                    model_uri = f"models:/{MODEL_NAME}/latest"
                    model_version = "latest"
            except Exception:
                # Model not registered yet, will retry
                raise Exception("Model not registered yet")
            
            logger.info(f"Loading model: {model_uri}")
            model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"Model loaded successfully (version: {model_version})")
            break
            
        except Exception as e:
            logger.warning(f"Model loading attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                logger.error("Failed to load model - will serve without predictions")
                model = None


def get_features_from_redis(symbol: str) -> Optional[Dict[str, float]]:
    """
    Fetch latest features from Redis.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
    
    Returns:
        Dictionary of feature values or None if not available
    """
    try:
        key = f"hft:features:{symbol.replace('/', '_')}"
        data = redis_client.hgetall(key)
        
        if not data:
            return None
        
        # Extract required features
        features = {}
        for col in FEATURE_COLUMNS:
            if col in data:
                features[col] = float(data[col])
            else:
                features[col] = 0.0
        
        return features
        
    except Exception as e:
        logger.error(f"Redis fetch error: {e}")
        return None


def make_prediction(features: Dict[str, float]) -> tuple:
    """
    Make prediction using loaded model.
    
    Args:
        features: Dictionary of feature values
    
    Returns:
        Tuple of (direction, confidence)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare feature array in correct order
    feature_array = np.array([[features.get(col, 0.0) for col in FEATURE_COLUMNS]])
    
    # Handle NaN/inf
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get prediction and probability
    prediction = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]
    confidence = float(max(probabilities))
    
    return int(prediction), confidence


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    redis_ok = False
    try:
        if redis_client:
            redis_client.ping()
            redis_ok = True
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model_version,
        redis_connected=redis_ok,
        predictions_count=prediction_count
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_from_features(request: PredictionRequest):
    """Make prediction from provided features."""
    global prediction_count, total_latency
    
    start_time = time.perf_counter()
    
    direction, confidence = make_prediction(request.features)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    prediction_count += 1
    total_latency += latency_ms
    
    # Prometheus metrics
    PREDICTION_COUNT.inc()
    LATENCY.observe(latency_ms / 1000.0)
    CONFIDENCE.observe(confidence)
    
    return PredictionResponse(
        symbol=SYMBOL,
        prediction="UP" if direction == 1 else "DOWN",
        direction=direction,
        confidence=confidence,
        features=request.features,
        latency_ms=latency_ms,
        model_version=model_version or "unknown",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/predict/live", response_model=PredictionResponse)
async def predict_live():
    """Make prediction using live features from Redis."""
    global prediction_count, total_latency
    
    start_time = time.perf_counter()
    
    # Fetch features from Redis
    features = get_features_from_redis(SYMBOL)
    
    if features is None:
        raise HTTPException(
            status_code=503,
            detail=f"Features not available for {SYMBOL}. Is Feature Calculator running?"
        )
    
    # Make prediction
    direction, confidence = make_prediction(features)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    prediction_count += 1
    total_latency += latency_ms
    
    # Log every 10 predictions
    if prediction_count % 10 == 0:
        avg_latency = total_latency / prediction_count
        logger.info(
            f"Prediction #{prediction_count} | "
            f"{'UP' if direction == 1 else 'DOWN'} ({confidence:.1%}) | "
            f"Latency: {latency_ms:.2f}ms (avg: {avg_latency:.2f}ms)"
        )
    
    return PredictionResponse(
        symbol=SYMBOL,
        prediction="UP" if direction == 1 else "DOWN",
        direction=direction,
        confidence=confidence,
        features=features,
        latency_ms=latency_ms,
        model_version=model_version or "unknown",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/metrics")
async def get_metrics():
    """Get model server metrics in Prometheus format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/features/{symbol}")
async def get_current_features(symbol: str):
    """Get current features for a symbol."""
    features = get_features_from_redis(symbol)
    
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"Features not found for {symbol}"
        )
    
    return {
        "symbol": symbol,
        "features": features,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
