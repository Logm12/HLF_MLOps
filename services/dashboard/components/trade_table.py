"""
Trade Table Component

Displays recent trades in a formatted table.
"""
import streamlit as st
from typing import List
import pandas as pd
from datetime import datetime

from clients import Trade


def render_trade_table(trades: List[Trade], max_rows: int = 20) -> None:
    """Render recent trades as a table."""
    if not trades:
        st.info("No trades executed yet")
        return
    
    # Convert to display format
    display_data = []
    for trade in trades[-max_rows:]:
        # Parse time
        try:
            time_str = trade.time.split("T")[1].split(".")[0] if "T" in trade.time else trade.time
        except Exception:
            time_str = trade.time[:8]
        
        pnl_str = f"${trade.pnl:.4f}" if trade.pnl is not None else "OPEN"
        pnl_display = "Profit" if trade.pnl and trade.pnl > 0 else "Loss" if trade.pnl and trade.pnl < 0 else "Pending"
        
        display_data.append({
            "Time": time_str,
            "Side": "BUY" if trade.side == "BUY" else "SELL",
            "Price": f"${trade.price:,.2f}",
            "Confidence": f"{trade.confidence:.1%}",
            "PnL": f"{pnl_display} {pnl_str}",
        })
    
    # Reverse to show newest first
    display_data.reverse()
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_trade_summary(trades: List[Trade]) -> None:
    """Render quick trade summary stats."""
    if not trades:
        return
    
    total_pnl = sum(t.pnl or 0 for t in trades)
    avg_confidence = sum(t.confidence for t in trades) / len(trades)
    buy_count = sum(1 for t in trades if t.side == "BUY")
    sell_count = len(trades) - buy_count
    
    cols = st.columns(4)
    cols[0].metric("Trades", len(trades))
    cols[1].metric("Buys/Sells", f"{buy_count}/{sell_count}")
    cols[2].metric("Avg Confidence", f"{avg_confidence:.1%}")
    cols[3].metric("Net PnL", f"${total_pnl:.4f}")
