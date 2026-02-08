# High-Frequency Trading (HFT) MLOps System

A robust, microservices-based High-Frequency Trading system implementing end-to-end MLOps practices. This project demonstrates an automated pipeline for data ingestion, feature engineering, model training, and low-latency inference using a modern Python technology stack.

## Key Features

*   **Low-Latency Architecture**: Optimized for sub-10ms inference latency using Redis and FastAPI.
*   **Real-time Streaming**: Apache Kafka backbone for high-throughput market data processing.
*   **Feature Store Implementation**: Feast (backed by Redis) ensures training-serving consistency.
*   **Hybrid AI Model**: Ensemble of XGBoost, LightGBM, Random Forest, and LSTM for robust prediction.
*   **Full Observability**: Integrated Prometheus and Grafana for system monitoring, with Streamlit for trading analytics.

## Technology Stack

*   **Core**: Python 3.10+, Docker, Docker Compose
*   **Streaming & Storage**: Apache Kafka, TimescaleDB, Redis
*   **ML & MLOps**: PyTorch, XGBoost, MLflow, Optuna, Feast
*   **API & UI**: FastAPI, Streamlit, Grafana, Prometheus

## Getting Started

### Prerequisites

*   Docker and Docker Compose installed on your machine.
*   Git for version control.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Logm12/HFT_great.git
    cd HFT_great
    ```

2.  **Start the Services**
    ```bash
    docker-compose up -d --build
    ```
    *Note: The build utilizes CPU-only PyTorch wheels to minimize image size.*

3.  **Access the Applications**
    *   **Trading Dashboard**: http://localhost:8501
    *   **Grafana Monitoring**: http://localhost:3001 (Credentials: `admin`/`admin`)
    *   **Prometheus**: http://localhost:9090
    *   **MLflow Registry**: http://localhost:5000

## Development

To run the test suite:
```bash
python -m unittest discover
```

To view service logs:
```bash
docker-compose logs -f trading-bot
```

## System Architecture Diagram

<p align="center">
  <img src="./assets/HFT_archtechture_diagrams.png" alt="Architechture Diagram" width="100%">
</p>


## How It Works

The system operates as a continuous loop of data ingestion, processing, and decision-making:

1.  **Market Data Ingestion**: The **Market Producer** maintains a persistent WebSocket connection to the Binance Exchange, receiving real-time trade execution data (ticks). Every new tick is immediately pushed to a dedicated topic in **Apache Kafka**.

2.  **Streaming & Feature Calculation**: **Kafka** streams these ticks to the **Feature Calculator** service. This service processes the raw price and volume data on-the-fly to compute technical indicators (like RSI and MACD).

3.  **Dual-Storage Strategy**:
    *   **Hot Storage (Redis)**: The computed features are written to **Redis**, which acts as an ultra-low latency online store. This ensures the trading bot always has access to the most recent market state (sub-millisecond access).
    *   **Cold Storage (TimescaleDB)**: Simultaneously, the **Data Ingestor** saves the raw tick data to **TimescaleDB** for long-term archival and future model retraining.

4.  **Inference & Execution**:
    *   The **Trading Bot** constantly polls the market state. When a potential opportunity arises, it sends a prediction request to the **Model Server**.
    *   The Model Server fetches the latest feature vector directly from **Redis**, runs it through the ensemble model, and returns a confidence score.
    *   If the score exceeds a threshold, the bot executes a buy or sell order back to Binance (currently simulated).

## Architecture

The system consists of the following isolated services:
1.  **Market Producer**: Ingests live WebSocket data.
2.  **Data Ingestor**: Persists historical data to TimescaleDB.
3.  **Feature Calculator**: Computes real-time technical indicators.
4.  **Model Trainer**: periodic offline model training and registry.
5.  **Model Server**: Serves real-time inference requests.
6.  **Trading Bot**: Simulates order execution based on model signals.

## License

This project is licensed under the MIT License.
