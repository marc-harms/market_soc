"""
SOC Market Seismograph - Detail Panel UI Components
===================================================

Deep dive analysis UI components for individual asset analysis.

Contains:
- render_regime_persistence_chart(): Horizontal bar chart showing regime duration
- render_current_regime_outlook(): Historical performance metrics for current regime
- render_detail_panel(): Main detailed analysis panel with charts and metrics

Author: Market Analysis Team
Version: 7.0 (Modularized)
"""

from typing import Dict, Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from logic import DataFetcher, SOCAnalyzer
from auth_manager import add_asset_to_portfolio, remove_asset_from_portfolio, get_current_user_id, get_user_portfolio


def render_regime_persistence_chart(current_regime: str, current_duration: int, regime_stats: Dict[str, Any], is_dark: bool = False) -> None:
    """
    Render a horizontal bar chart showing current regime duration vs historical average.
    
    Args:
        current_regime: Name of current regime (e.g., 'STABLE')
        current_duration: Days in current regime
        regime_stats: Historical statistics for this regime
        is_dark: Dark mode flag
    """
    # Get historical stats (using correct keys from logic.py)
    mean_duration = regime_stats.get('avg_duration', 0)  # avg_duration, not mean_duration
    median_duration = regime_stats.get('median_duration', 0)
    max_duration = regime_stats.get('max_duration', 0)
    p95_duration = regime_stats.get('p95_duration', 0)
    
    # Regime colors
    regime_colors = {
        'STABLE': '#00C864',
        'ACTIVE': '#FFCC00',
        'HIGH_ENERGY': '#FF6600',
        'CRITICAL': '#FF4040',
        'DORMANT': '#888888'
    }
    
    regime_color = regime_colors.get(current_regime, '#667eea')
    
    # Handle edge cases
    if max_duration == 0:
        max_duration = max(current_duration * 2, 30)  # Fallback
    if mean_duration == 0:
        mean_duration = current_duration  # Use current as reference
    
    # Theme-aware colors
    text_color = '#FFFFFF' if is_dark else '#1a1a1a'
    axis_color = '#CCCCCC' if is_dark else '#333333'
    grid_color = '#444444' if is_dark else '#E0E0E0'
    bg_color = 'rgba(0,0,0,0)' if is_dark else 'rgba(248,248,248,1)'
    annotation_color = '#FFFFFF' if is_dark else '#333333'
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Background range (0 to max)
    fig.add_trace(go.Bar(
        y=['Duration'],
        x=[max_duration],
        orientation='h',
        marker=dict(color='rgba(200,200,200,0.3)' if is_dark else 'rgba(200,200,200,0.5)'),
        name='Max Observed',
        showlegend=False
    ))
    
    # Current duration bar
    fig.add_trace(go.Bar(
        y=['Duration'],
        x=[current_duration],
        orientation='h',
        marker=dict(color=regime_color),
        name='Current',
        showlegend=False
    ))
    
    # Add vertical lines for mean and P95
    fig.add_vline(x=mean_duration, line_dash="dash", line_color="#667eea", line_width=3,
                  annotation_text=f"Avg: {mean_duration:.0f}d", annotation_position="top",
                  annotation=dict(font=dict(color=annotation_color, size=13)))
    
    if p95_duration > 0:
        fig.add_vline(x=p95_duration, line_dash="dot", line_color="#FF6600", line_width=3,
                      annotation_text=f"95th: {p95_duration:.0f}d", annotation_position="bottom",
                      annotation=dict(font=dict(color=annotation_color, size=13)))
    
    # Update layout with explicit colors
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor='rgba(0,0,0,0)' if is_dark else 'rgba(255,255,255,0)',
        plot_bgcolor=bg_color,
        height=180,
        margin=dict(l=80, r=30, t=50, b=50),
        showlegend=False,
        font=dict(color=text_color, size=13)
    )
    
    # Update axes separately (correct Plotly API)
    fig.update_xaxes(
        range=[0, max_duration * 1.1],
        title_text="Days",
        title_font=dict(color=axis_color, size=14),
        tickfont=dict(color=axis_color, size=12),
        gridcolor=grid_color
    )
    
    fig.update_yaxes(
        title_text="",
        tickfont=dict(color=axis_color, size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    if current_duration > p95_duration:
        interpretation = f"⚠️ **Statistical Anomaly:** This regime has lasted {current_duration} days, which is unusually long (above 95th percentile of {p95_duration:.0f} days). Mean reversion probability is elevated."
        st.warning(interpretation)
    elif current_duration > mean_duration:
        interpretation = f"📊 This regime has lasted {current_duration} days, which is **above** the historical average of {mean_duration:.0f} days. Median duration: {median_duration:.0f} days."
        st.info(interpretation)
    else:
        interpretation = f"📊 This regime is still relatively young at {current_duration} days, **below** the historical average of {mean_duration:.0f} days. Median duration: {median_duration:.0f} days."
        st.info(interpretation)


def render_current_regime_outlook(current_regime: str, regime_data: Dict[str, Any]) -> None:
    """
    Render a table showing the historical outlook for the CURRENT regime only.
    
    Args:
        current_regime: Name of current regime
        regime_data: Statistical data for this regime
    """
    regime_display = current_regime.replace('_', ' ').title()
    regime_emojis = {'STABLE': '🟢', 'ACTIVE': '🟡', 'HIGH_ENERGY': '🟠', 'CRITICAL': '🔴', 'DORMANT': '⚪'}
    emoji = regime_emojis.get(current_regime, '📊')
    
    st.markdown(f"##### 🎯 Historical Outlook: {emoji} {regime_display} Regime")
    st.caption("📊 **How to read this:** Shows average price movements following the start of this regime in the past. Use this to understand typical behavior patterns for the current market state.")
    
    # Check if we have data
    phase_count = regime_data.get('phase_count', 0)
    if phase_count == 0:
        st.info("No historical data available for this regime.")
        return
    
    st.markdown(f"*Based on **{phase_count}** historical occurrences of this regime*")
    
    # Build outlook table using available data
    ret_10d = regime_data.get('start_return_10d', 0)
    ret_30d = regime_data.get('avg_return_30d', 0)
    ret_90d = regime_data.get('avg_return_90d', 0)
    dd_10d = regime_data.get('worst_max_dd_10d', 0)
    avg_price_change = regime_data.get('avg_price_change_during', 0)
    
    # Create metrics in columns for better display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("10-Day Avg Return", f"{ret_10d:+.1f}%")
    with col2:
        st.metric("30-Day Avg Return", f"{ret_30d:+.1f}%")
    with col3:
        st.metric("90-Day Avg Return", f"{ret_90d:+.1f}%")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric("Avg During Phase", f"{avg_price_change:+.1f}%")
    with col5:
        st.metric("Worst 10d Drawdown", f"{dd_10d:.1f}%" if dd_10d != 0 else "N/A")
    with col6:
        st.metric("Phase Count", f"{phase_count}")
    
    # Add interpretation
    if ret_30d > 5:
        st.success(f"📈 Historically, this regime has shown **positive momentum** with an average 30-day return of {ret_30d:+.1f}%.")
    elif ret_30d < -5:
        st.error(f"📉 Historically, this regime has shown **negative momentum** with an average 30-day return of {ret_30d:+.1f}%.")
    elif ret_30d != 0:
        st.info(f"📊 Historically, this regime has shown **neutral momentum** with an average 30-day return of {ret_30d:+.1f}%.")
    else:
        st.info("📊 Insufficient historical data for return analysis.")


def render_detail_panel(result: Dict[str, Any], get_signal_color_func, get_signal_bg_func) -> None:
    """
    Render detailed analysis panel for a selected asset.
    
    Simplified, condensed view:
    - Hero card with core facts (name, price, trend, regime, criticality)
    - SOC analysis plot
    
    Args:
        result: Dictionary containing asset analysis results
        get_signal_color_func: Function to get signal color
        get_signal_bg_func: Function to get signal background color
    """
    is_dark = st.session_state.get('dark_mode', True)
    symbol = result['symbol']
    signal = result.get('signal', 'Unknown')
    price = result.get('price', 0.0)
    trend = result.get('trend', 'Unknown')
    criticality = int(result.get('criticality_score', 0))
    name = result.get('name', symbol)
    full_name = result.get('info', {}).get('longName', name)
    
    # Determine traffic-light color for regime status
    signal_lower = signal.lower()
    if "critical" in signal_lower:
        regime_color = "#C0392B"
        regime_label = "Critical"
    elif "high" in signal_lower:
        regime_color = "#D35400"
        regime_label = "High Energy"
    elif "active" in signal_lower:
        regime_color = "#F1C40F"
        regime_label = "Active"
    elif "stable" in signal_lower:
        regime_color = "#27AE60"
        regime_label = "Stable"
    else:
        regime_color = "#95A5A6"
        regime_label = signal
    
    # Hero Card
    st.markdown(f"""
    <div style="
        border: 1px solid #D1C4E9;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 16px;
        margin: 8px 0 16px 0;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.06);
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap;">
            <div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #2C3E50;">{name}</div>
                <div style="font-size: 0.95rem; color: #555; margin-top: 4px;">{full_name}</div>
                <div style="font-size: 1.2rem; font-weight: 700; margin-top: 8px; color: #2C3E50;">${price:,.2f}</div>
                <div style="font-size: 0.95rem; color: #666;">Trend: <strong>{trend}</strong></div>
            </div>
            <div style="text-align: right; min-width: 180px;">
                <div style="font-size: 0.95rem; color: #666; margin-bottom: 6px;">Regime Status</div>
                <div style="
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    background: rgba(0,0,0,0.02);
                    border: 1px solid {regime_color};
                    border-radius: 999px;
                    padding: 6px 12px;
                ">
                    <span style="
                        display: inline-block;
                        width: 14px;
                        height: 14px;
                        border-radius: 50%;
                        background: {regime_color};
                        border: 1px solid rgba(0,0,0,0.1);
                    "></span>
                    <span style="font-weight: 700; color: #2C3E50;">{regime_label}</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.95rem; color: #666;">Criticality</div>
                <div style="
                    font-size: 1.3rem;
                    font-weight: 800;
                    color: {regime_color};
                    background: rgba(0,0,0,0.02);
                    border: 1px solid {regime_color};
                    border-radius: 6px;
                    padding: 6px 10px;
                    display: inline-block;
                    min-width: 120px;
                ">{criticality}/100</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # SOC Chart only (condensed output)
    fetcher = DataFetcher(cache_enabled=True)
    df = fetcher.fetch_data(symbol)
    if not df.empty:
        analyzer = SOCAnalyzer(df, symbol, result.get('info'))
        figs = analyzer.get_plotly_figures(dark_mode=is_dark)
        st.plotly_chart(figs['chart3'], width="stretch")
    else:
        st.warning("No data available for this asset.")

