"""
Model Server API Client

Fetches predictions and model health from Model Server.
"""
import logging
from typing import Optional, Dict
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Model prediction result."""
    symbol: str
    prediction: str  # UP or DOWN
    direction: int
    confidence: float
    latency_ms: float
    model_version: str
    features: Dict[str, float]


@dataclass
class ModelHealth:
    """Model server health status."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    redis_connected: bool
    predictions_count: int


class ModelClient:
    """Client for Model Server API."""
    
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def get_prediction(self) -> Optional[Prediction]:
        """Fetch live prediction."""
        try:
            response = self._client.get(f"{self.base_url}/predict/live")
            if response.status_code == 200:
                data = response.json()
                return Prediction(
                    symbol=data.get('symbol', ''),
                    prediction=data.get('prediction', ''),
                    direction=data.get('direction', 0),
                    confidence=data.get('confidence', 0),
                    latency_ms=data.get('latency_ms', 0),
                    model_version=data.get('model_version', ''),
                    features=data.get('features', {}),
                )
        except Exception as e:
            logger.error(f"Failed to fetch prediction: {e}")
        return None
    
    def get_health(self) -> Optional[ModelHealth]:
        """Fetch model server health."""
        try:
            response = self._client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                return ModelHealth(
                    status=data.get('status', 'unknown'),
                    model_loaded=data.get('model_loaded', False),
                    model_version=data.get('model_version'),
                    redis_connected=data.get('redis_connected', False),
                    predictions_count=data.get('predictions_count', 0),
                )
        except Exception as e:
            logger.error(f"Failed to fetch health: {e}")
        return None
    
    def is_healthy(self) -> bool:
        """Check if model server is healthy."""
        health = self.get_health()
        return health is not None and health.model_loaded
