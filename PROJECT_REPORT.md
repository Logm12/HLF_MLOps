# Project Technical Report
# High-Frequency Trading (HFT) MLOps System

## 1. Executive Summary

This report details the design, implementation, and evaluation of a High-Frequency Trading (HFT) system integrated with a comprehensive MLOps pipeline. The project addresses the challenge of adapting algorithmic trading strategies to rapid market shifts by automating the end-to-end workflow from data ingestion to model deployment.

The solution leverages a microservices architecture, implementing a Feature Store (Feast) for low-latency feature retrieval and a hybrid Ensemble Model (XGBoost, LightGBM, Random Forest, LSTM) for robust prediction. Key achievements include an average end-to-end system latency of under 10ms and a directional accuracy of 51.7% on real-world Binance market data.

## 2. System Architecture

The system follows a modular microservices architecture designed for scalability, maintainability, and fault tolerance.

<p align="center">
  <img src="./assets/HFT_archtechture_diagrams.png" alt="Architechture Diagram" width="100%">
</p>

### 2.1. System Flow Overview

The architecture facilitates a high-speed data pipeline tailored for algorithmic trading:

1.  **Market Data Socket**: The system connects to the Binance WebSocket API via the **Market Producer**, ensuring real-time receipt of trade ticks. This data is immediately published to **Apache Kafka**.
2.  **Stream Processing**: The **Feature Calculator** service consumes the Kafka stream. It calculates technical indicators (RSI, Moving Averages) on the fly and pushes the results to **Redis** (the Online Feature Store).
3.  **Data Persistence**: A separate **Data Ingestor** service writes the raw tick data to **TimescaleDB**. This creates a historical record used for offline model training and backtesting.
4.  **Signal Generation**: The **Trading Bot** continuously monitors market conditions. Upon identifying a potential trade setup, it queries the **Model Server**, which retrieves the latest features from Redis and returns a prediction confidence score.
5.  **Execution and Monitoring**: High-confidence predictions trigger mock orders. All system metrics (latency, memory usage) are scraped by **Prometheus** and visualized on **Grafana**, while trading performance is tracked on the **Streamlit** dashboard.

### 2.2. Architectural Components

*   **Data Layer**:
    *   **TimescaleDB**: Used for persistent storage of historical market data (ticks), optimized for time-series queries.
    *   **Redis**: Serves as the Online Feature Store, providing sub-millisecond access to pre-computed feature vectors during inference.
*   **Streaming Layer**:
    *   **Apache Kafka**: Acts as the central message backbone, decoupling data producers from consumers and ensuring high-throughput handling of market ticks.
*   **Computation Layer**:
    *   **Feature Calculator**: A stream processing service that consumes raw ticks from Kafka, computes technical indicators (RSI, MACD, etc.) in real-time, and updates the Feature Store.
    *   **Model Server**: A high-performance inference engine built with FastAPI that serves predictions upon request.
*   **MLOps Layer**:
    *   **MLflow**: Manages the machine learning lifecycle, including experiment tracking, model versioning, and artifact storage.
    *   **Model Trainer**: An offline service that periodically retrains models using historical data and registers the best-performing versions.

### 2.2. Data Flow

1.  **Ingestion**: The Market Producer connects to the Binance WebSocket API and pushes raw tick data to the `market_ticks` Kafka topic.
2.  **Processing**: The Feature Calculator subscribes to `market_ticks`, computes technical indicators, and persists the latest values to Redis (Online Store) and historical records to TimescaleDB (Offline Store).
3.  **Inference**: The Trading Bot triggers a prediction request. The Model Server fetches the latest feature vector from Redis and returns a prediction confidence score.
4.  **Execution**: Based on the prediction confidence, the Trading Bot executes a simulated order and logs the result.

## 3. Technology Stack

The technology selection prioritizes lightweight, high-performance tools suitable for a Python-centric ecosystem.

*   **Programming Language**: Python 3.10+ (Utilizing `asyncio` for non-blocking I/O).
*   **Message Broker**: Apache Kafka (Industry standard for reliable data streaming).
*   **Feature Store**: Feast (Ensures consistency between training and serving features).
*   **Database**: TimescaleDB (SQL-based time-series management).
*   **Model Serving**: FastAPI + MLflow.
*   **Containerization**: Docker & Docker Compose (Ensures reproducible environments).

## 4. Algorithmic Approach

To address the noisy nature of high-frequency cryptocurrency data, a Hybrid Ensemble method was employed.

*   **Gradient Boosting (XGBoost & LightGBM)**: Effective at capturing non-linear relationships in tabular market data.
*   **Bagging (Random Forest)**: Reduces model variance and mitigates overfitting.
*   **Deep Learning (LSTM)**: Captures temporal dependencies and sequence patterns in price movements.
*   **Optimization**: Hyperparameters are tuned using Optuna to maximize the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

## 5. Implementation Details

### 5.1. Performance Optimization
*   **Docker Optimization**: Multi-stage builds and CPU-only PyTorch binaries reduced image sizes significantly (e.g., PyTorch footprint reduced from ~2GB to ~200MB).
*   **Latency Management**: Network serialization overhead was minimized by using Protocol Buffers and optimized internal Docker networking.

### 5.2. Observability
*   **System Metrics**: Prometheus scrapes CPU, memory, and request latency metrics from all services.
*   **Visualization**: Grafana provides a centralized dashboard for system health monitoring.
*   **Business Metrics**: A Streamlit dashboard displays real-time trading performance, including Profit and Loss (PnL), Win Rate, and Order History.

## 6. Evaluation Results

The system was validated using a 2-year historical dataset (BTC/USDT 15-minute intervals).

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Accuracy** | 51.7% | Directional prediction accuracy on out-of-sample data. |
| **Latency** | 4-8 ms | Average end-to-end processing time from tick arrival to signal generation. |
| **Throughput** | >1000 ticks/s | System capacity for processing incoming market data. |
| **Uptime** | 99.9% | Observed availability during 24-hour continuous testing. |

## 7. Future Work

*   **Live Execution**: Transition from reliable simulation to live order execution via Exchange APIs.
*   **Advanced Features**: Integration of sentiment analysis features derived from news and social media streams.
*   **Orchestration**: Migration to Kubernetes for automated scaling and advanced deployment strategies (Canary/Blue-Green).
