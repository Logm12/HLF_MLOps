"""
HFT Trading Dashboard

Real-time monitoring dashboard for the trading system.
Entry point - wires together clients and components.
"""
import streamlit as st
import time
import logging

import config
from clients import TradingClient, ModelClient
from components import (
    render_trading_kpis, render_latency_kpis, render_position_card,
    render_pnl_chart, render_trade_distribution, render_confidence_gauge,
    render_trade_table, render_trade_summary,
    render_recommendations,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="HFT Trading Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better appearance
# Custom CSS for better appearance
st.markdown("""
<style>
    /* Card style for metrics */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color, #f0f2f6);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid var(--text-color-20, rgba(49, 51, 63, 0.2));
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Improve spacing */
    .block-container {
        padding-top: 1.5rem;
    }
    
    /* Better table styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--text-color-20, rgba(49, 51, 63, 0.2));
        border-radius: 5px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_clients():
    """Initialize API clients (cached)."""
    trading_client = TradingClient(config.TRADING_BOT_URL)
    model_client = ModelClient(config.MODEL_SERVER_URL)
    return trading_client, model_client


def main():
    """Main dashboard entry point."""
    # Header
    st.title("HFT Trading Dashboard")
    st.caption(f"Symbol: {config.SYMBOL} | Refresh: {config.REFRESH_TRADES}s")
    
    # Get clients
    trading_client, model_client = get_clients()
    
    # Sidebar - refresh controls
    with st.sidebar:
        st.header("Settings")
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_rate = st.slider("Refresh (sec)", 1, 30, config.REFRESH_TRADES)
        
        st.divider()
        st.subheader("Quick Links")
        st.markdown(f"- [MLflow UI](http://localhost:5000)")
        st.markdown(f"- [Model API](http://localhost:8001/docs)")
        st.markdown(f"- [Kafka UI](http://localhost:8085)")
    
    # Fetch data
    stats = trading_client.get_stats()
    trades = trading_client.get_trades(limit=config.MAX_TRADES_DISPLAY)
    position = trading_client.get_position()
    prediction = model_client.get_prediction()
    
    # Connection status
    if stats is None or prediction is None:
        st.error("Unable to connect to services. Please check if Trading Bot and Model Server are running.")
    
    # Trading Performance KPIs
    st.subheader("Trading Performance")
    render_trading_kpis(stats)
    
    # Latency & Model KPIs
    st.subheader("System Performance")
    render_latency_kpis(stats, prediction)
    
    # Charts and Position
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("PnL Over Time")
        render_pnl_chart(trades)
    
    with col2:
        st.subheader("Current Status")
        if position:
            render_position_card(position.position, position.entry_price, prediction)
        else:
            st.info("Position data unavailable")
        
        st.divider()
        if prediction:
            render_confidence_gauge(prediction.confidence)
    
    # Trade Table
    st.divider()
    st.subheader("Recent Trades")
    render_trade_summary(trades)
    render_trade_table(trades, max_rows=config.MAX_TRADES_DISPLAY)
    
    # Model Recommendations
    st.divider()
    render_recommendations(stats, prediction)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
