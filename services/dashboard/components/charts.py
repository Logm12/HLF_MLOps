"""
Charts Component

Simple charts for PnL and trade visualization.
"""
import streamlit as st
from typing import List
import pandas as pd

from clients import Trade, TradingStats


def render_pnl_chart(trades: List[Trade]) -> None:
    """Render cumulative PnL line chart."""
    if not trades:
        st.info("No trades yet")
        return
    
    # Calculate cumulative PnL
    pnl_data = []
    cumulative = 0.0
    
    for trade in trades:
        if trade.pnl is not None:
            cumulative += trade.pnl
        pnl_data.append({
            "Trade #": len(pnl_data) + 1,
            "PnL ($)": cumulative,
        })
    
    if not pnl_data:
        st.info("No completed trades yet")
        return
    
    df = pd.DataFrame(pnl_data)
    st.line_chart(df.set_index("Trade #"), height=250)


def render_trade_distribution(stats: TradingStats) -> None:
    """Render win/loss bar chart."""
    if stats.total_trades == 0:
        st.info("No trades yet")
        return
    
    data = pd.DataFrame({
        "Type": ["Winning", "Losing"],
        "Count": [stats.winning_trades, stats.losing_trades],
    })
    
    st.bar_chart(data.set_index("Type"), height=200)


def render_confidence_gauge(confidence: float) -> None:
    """Render confidence level as progress bar (Streamlit native)."""
    # Streamlit doesn't have gauge, use progress bar
    st.progress(confidence, text=f"Model Confidence: {confidence:.1%}")
    
    # Color indicator
    if confidence >= 0.60:
        st.caption("🟢 High confidence")
    elif confidence >= 0.55:
        st.caption("🟡 Moderate confidence")
    else:
        st.caption("🔴 Low confidence - trade may be skipped")
