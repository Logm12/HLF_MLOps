"""
KPI Cards Component

Displays key performance indicators as metric cards.
"""
import streamlit as st
from typing import Optional

from clients import TradingStats, Prediction


def render_trading_kpis(stats: Optional[TradingStats]) -> None:
    """Render trading performance KPIs."""
    if stats is None:
        st.warning("Trading data unavailable")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pnl_color = "normal" if stats.total_pnl >= 0 else "inverse"
        st.metric(
            label="Total PnL",
            value=f"${stats.total_pnl:.4f}",
            delta=f"{stats.total_trades} trades",
            delta_color=pnl_color,
        )
    
    with col2:
        win_delta = "Pass" if stats.win_rate >= 50 else "Low"
        st.metric(
            label="Win Rate",
            value=f"{stats.win_rate:.1f}%",
            delta=f"{stats.winning_trades}W / {stats.losing_trades}L",
        )
    
    with col3:
        st.metric(
            label="Max Drawdown",
            value=f"${stats.max_drawdown:.4f}",
            delta="From peak",
            delta_color="inverse",
        )
    
    with col4:
        pf = stats.profit_factor
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        st.metric(
            label="Profit Factor",
            value=pf_str,
            delta=f"${stats.avg_trade_pnl:.4f}/trade",
        )


def render_latency_kpis(stats: Optional[TradingStats], prediction: Optional[Prediction]) -> None:
    """Render latency-related KPIs."""
    col1, col2, col3 = st.columns(3)
    
    avg_latency = stats.avg_latency_ms if stats else 0
    pred_latency = prediction.latency_ms if prediction else 0
    total_predictions = stats.predictions_count if stats else 0
    
    with col1:
    with col1:
        latency_status = "Good" if avg_latency < 10 else "Fair" if avg_latency < 50 else "Poor"
        st.metric(
            label=f"{latency_status} Avg Latency",
            value=f"{avg_latency:.2f} ms",
            delta=f"Last: {pred_latency:.2f} ms",
        )
    
    with col2:
        st.metric(
            label="Total Predictions",
            value=f"{total_predictions:,}",
        )
    
    with col3:
        confidence = prediction.confidence if prediction else 0
        conf_color = "normal" if confidence >= 0.55 else "off"
        st.metric(
            label="Confidence",
            value=f"{confidence:.1%}",
            delta=prediction.prediction if prediction else "N/A",
            delta_color=conf_color,
        )


def render_position_card(position_str: str, entry_price: Optional[float], prediction: Optional[Prediction]) -> None:
    """Render current position and prediction."""
    col1, col2 = st.columns(2)
    
    with col1:
        if position_str == "LONG":
            st.success(f"LONG @ ${entry_price:,.2f}" if entry_price else "LONG")
        elif position_str == "SHORT":
            st.error(f"SHORT @ ${entry_price:,.2f}" if entry_price else "SHORT")
        else:
            st.info("FLAT - No position")
    
    with col2:
        if prediction:
            if prediction.prediction == "UP":
                st.success(f"Prediction: **UP** ({prediction.confidence:.1%})")
            else:
                st.error(f"Prediction: **DOWN** ({prediction.confidence:.1%})")
        else:
            st.warning("No prediction available")
