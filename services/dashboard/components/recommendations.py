"""
Model Recommendations Component

Provides dynamic recommendations based on current metrics.
"""
import streamlit as st
from typing import List, Dict, Optional
from dataclasses import dataclass

from clients import TradingStats, Prediction
import config


@dataclass
class Recommendation:
    """Model improvement recommendation."""
    priority: str  # HIGH, MEDIUM, LOW
    issue: str
    solutions: List[str]


def analyze_model_performance(
    stats: Optional[TradingStats],
    prediction: Optional[Prediction],
) -> List[Recommendation]:
    """Generate recommendations based on current metrics."""
    recommendations = []
    
    if stats is None or prediction is None:
        recommendations.append(Recommendation(
            priority="HIGH",
            issue="Services unavailable",
            solutions=["Check Trading Bot and Model Server status"]
        ))
        return recommendations
    
    thresholds = config.THRESHOLDS
    
    # Win rate analysis
    if stats.win_rate < thresholds["win_rate_low"] * 100:
        recommendations.append(Recommendation(
            priority="HIGH",
            issue=f"Low win rate ({stats.win_rate:.1f}%)",
            solutions=[
                "Increase confidence threshold from 0.55 to 0.60",
                "Add momentum confirmation (RSI + MACD alignment)",
                "Consider longer prediction horizon (5min instead of 1min)",
                "Add volume spike filter to avoid noise trades",
            ]
        ))
    
    # Confidence analysis
    avg_confidence = prediction.confidence
    if avg_confidence < thresholds["confidence_low"]:
        recommendations.append(Recommendation(
            priority="MEDIUM",
            issue=f"Low model confidence ({avg_confidence:.1%})",
            solutions=[
                "Retrain with more data (>20,000 samples)",
                "Add feature engineering: ATR ratio, VWAP deviation",
                "Try ensemble: XGBoost + LightGBM voting",
                "Use hyperparameter tuning (Optuna/Ray Tune)",
            ]
        ))
    
    # Drawdown analysis
    if stats.max_drawdown > thresholds["drawdown_high"]:
        recommendations.append(Recommendation(
            priority="HIGH",
            issue=f"High drawdown (${stats.max_drawdown:.4f})",
            solutions=[
                "Implement stop-loss at 2% of position",
                "Reduce position size during low confidence",
                "Add trailing stop to protect profits",
                "Consider max daily loss limit",
            ]
        ))
    
    # Latency analysis
    if stats.avg_latency_ms > thresholds["latency_high"]:
        recommendations.append(Recommendation(
            priority="MEDIUM",
            issue=f"High latency ({stats.avg_latency_ms:.1f}ms)",
            solutions=[
                "Optimize feature calculation (vectorize)",
                "Use model quantization for faster inference",
                "Consider batching predictions",
            ]
        ))
    
    # Profit factor
    if stats.profit_factor < 1.2 and stats.total_trades > 5:
        recommendations.append(Recommendation(
            priority="MEDIUM",
            issue=f"Low profit factor ({stats.profit_factor:.2f})",
            solutions=[
                "Increase take-profit targets",
                "Filter trades during high volatility hours",
                "Add trade validation with multiple timeframes",
            ]
        ))
    
    # If everything looks good
    if not recommendations:
        recommendations.append(Recommendation(
            priority="LOW",
            issue="Model performing well",
            solutions=[
                "Continue monitoring for market regime changes",
                "Consider adding more trading pairs",
                "Document current configuration for reference",
            ]
        ))
    
    return recommendations


def render_recommendations(
    stats: Optional[TradingStats],
    prediction: Optional[Prediction],
) -> None:
    """Render model recommendations panel."""
    st.subheader("Model Improvement Recommendations")
    
    recommendations = analyze_model_performance(stats, prediction)
    
    for rec in recommendations:
        # Color based on priority
        if rec.priority == "HIGH":
            expander_title = f"HIGH: {rec.issue}"
            container = st.expander(expander_title, expanded=True)
        elif rec.priority == "MEDIUM":
            expander_title = f"MEDIUM: {rec.issue}"
            container = st.expander(expander_title, expanded=False)
        else:
            expander_title = f"LOW: {rec.issue}"
            container = st.expander(expander_title, expanded=False)
        
        with container:
            st.markdown("**Suggested Actions:**")
            for i, solution in enumerate(rec.solutions, 1):
                st.markdown(f"{i}. {solution}")
