
"""
SOC Market Seismograph - Streamlit Application
==============================================

Interactive web dashboard for Self-Organized Criticality (SOC) market analysis.

Features:
    - Multi-asset scanning with 5-Tier regime classification
    - Deep dive analysis with historical signal performance
    - Instability Score indicator (0-100)
    - Lump Sum investment simulation with Dynamic Position Sizing
    - Dark/Light theme support

Theory:
    Markets exhibit Self-Organized Criticality - they naturally evolve toward
    critical states where small inputs can trigger events of any size.
    This app visualizes market "energy states" through volatility clustering.

Author: Market Analysis Team
Version: 6.0 (Cleaned & Documented)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import requests
from typing import List, Dict, Any

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from logic import DataFetcher, SOCAnalyzer, run_dca_simulation, calculate_audit_metrics
from ui_simulation import render_dca_simulation
from ui_detail import render_detail_panel, render_regime_persistence_chart, render_current_regime_outlook
from ui_auth import render_disclaimer, render_auth_page, render_sticky_cockpit_header, render_education_landing
from auth_manager import (
    is_authenticated, logout, get_current_user_id, get_current_user_email,
    get_user_portfolio, can_access_simulation, show_upgrade_prompt,
    can_run_simulation, increment_simulation_count
)
from config import get_scientific_heritage_css, HERITAGE_THEME, REGIME_COLORS


# =============================================================================
# ADVANCED ANALYTICS (Simplified: Frequency + Duration Only)
# =============================================================================
def render_advanced_analytics(df: pd.DataFrame, is_dark: bool = False) -> None:
    """
    Render simplified advanced regime analytics: Frequency and Duration only.
    
    Layout:
    - Column 1: Frequency Analysis (Donut chart + table)
    - Column 2: Duration Analysis (Box plot + table)
    
    Args:
        df: Price dataframe with Close or Adj Close column
        is_dark: Dark mode flag
    """
    if df is None or df.empty:
        st.info("No data available for advanced analytics.")
        return
    
    regimes_order = ['DORMANT', 'STABLE', 'ACTIVE', 'HIGH_ENERGY', 'CRITICAL']
    colors = {
        'DORMANT': REGIME_COLORS.get('DORMANT', '#95A5A6'),
        'STABLE': REGIME_COLORS.get('STABLE', '#27AE60'),
        'ACTIVE': REGIME_COLORS.get('ACTIVE', '#F1C40F'),
        'HIGH_ENERGY': REGIME_COLORS.get('HIGH_ENERGY', '#D35400'),
        'CRITICAL': REGIME_COLORS.get('CRITICAL', '#C0392B'),
    }
    text_color = "#FFFFFF" if is_dark else "#333333"
    
    # Derive regimes from volatility of returns (simple heuristic)
    df_local = df.copy()
    # Pick a price series robustly
    if 'Close' in df_local.columns:
        price_series = df_local['Close']
    elif 'Adj Close' in df_local.columns:
        price_series = df_local['Adj Close']
    else:
        # try first numeric column
        num_cols = df_local.select_dtypes(include='number').columns
        if len(num_cols) == 0:
            st.info("No price data available for advanced analytics.")
            return
        price_series = df_local[num_cols[0]]
    
    df_local['return'] = price_series.pct_change()
    df_local['vol'] = df_local['return'].abs()
    df_local['Regime'] = pd.cut(
        df_local['vol'],
        bins=[-1, 0.005, 0.01, 0.02, 0.03, 10],
        labels=['DORMANT', 'STABLE', 'ACTIVE', 'HIGH_ENERGY', 'CRITICAL']
    ).astype(str)
    
    # Run-length encoding for durations
    runs = []
    prev = None
    start_idx = None
    for idx, reg in df_local['Regime'].items():
        if reg != prev:
            if prev is not None:
                runs.append((prev, start_idx, idx))
            prev = reg
            start_idx = idx
    if prev is not None:
        runs.append((prev, start_idx, df_local.index[-1]))
    
    durations = pd.DataFrame(runs, columns=['Regime', 'Start', 'End'])
    durations['Duration'] = (durations['End'] - durations['Start']).dt.days.clip(lower=1)
    
    # Frequency summary
    freq_days = df_local['Regime'].value_counts().reindex(regimes_order).fillna(0)
    freq_runs = durations['Regime'].value_counts().reindex(regimes_order).fillna(0)
    total_days = freq_days.sum()
    freq_share = (freq_days / total_days * 100).replace([float('inf'), float('nan')], 0)
    
    freq_df = pd.DataFrame({
        'Total Days': freq_days.astype(int),
        'Share (%)': freq_share.round(1),
        'Occurrences': freq_runs.astype(int)
    })
    freq_df.index.name = "Regime"
    
    # Duration summary
    dur_stats = durations.groupby('Regime')['Duration'].agg(['min', 'mean', 'median', 'max']).reindex(regimes_order).fillna(0)
    dur_stats = dur_stats.rename(columns={
        'min': 'Min Days',
        'mean': 'Avg Days',
        'median': 'Median Days',
        'max': 'Max Days'
    })
    dur_stats['Avg Days'] = dur_stats['Avg Days'].round(1)
    dur_stats.index.name = "Regime"
    
    with st.expander("Advanced Analytics: Probability, Duration & Risk Models", expanded=False):
        col_freq, col_dur = st.columns(2)
        
        # === COLUMN 1: FREQUENCY ANALYSIS ===
        with col_freq:
            # Donut Chart
            fig_freq = go.Figure(data=[go.Pie(
                labels=freq_days.index,
                values=freq_days.values,
                hole=0.5,
                marker=dict(colors=[colors.get(r, '#888') for r in freq_days.index]),
                textinfo='label+percent'
            )])
            fig_freq.update_layout(
                title="Historical Regime Frequency",
                annotations=[dict(
                    text=f"{int(total_days)} Days",
                    showarrow=False,
                    font=dict(size=14, family="Merriweather, serif", color=text_color)
                )],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Merriweather, serif", color=text_color, size=11),
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
                height=350
            )
            st.plotly_chart(fig_freq, use_container_width=True)
            
            # Frequency Table
            st.markdown("**Frequency Breakdown:**")
            st.dataframe(freq_df, use_container_width=True)
        
        # === COLUMN 2: DURATION ANALYSIS ===
        with col_dur:
            # Box Plot
            fig_dur = go.Figure()
            for reg in regimes_order:
                data = durations.loc[durations['Regime'] == reg, 'Duration']
                if not data.empty:
                    fig_dur.add_trace(go.Box(
                        y=data,
                        name=reg,
                        boxpoints='outliers',
                        marker_color=colors.get(reg, '#888')
                    ))
            fig_dur.update_layout(
                title="Regime Duration Statistics (Days)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Merriweather, serif", color=text_color, size=11),
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
                height=350
            )
            fig_dur.update_yaxes(title_text="Days", gridcolor='#E6E1D3', gridwidth=0.5)
            st.plotly_chart(fig_dur, use_container_width=True)
            
            # Duration Table
            st.markdown("**Duration Statistics:**")
            st.dataframe(dur_stats, use_container_width=True)
        
        # === EVENT STUDY ANALYSIS ===
        st.markdown("---")
        st.markdown("### 📉 Event Study: Price Behavior Around Regime Changes")
        st.caption("*Analyze price movements before and after regime transitions*")
        
        # Identify regime change trigger events
        df_local['Prev_Regime'] = df_local['Regime'].shift(1)
        df_local['Regime_Changed'] = (df_local['Regime'] != df_local['Prev_Regime'])
        trigger_events = df_local[df_local['Regime_Changed']].copy()
        
        if len(trigger_events) < 2:
            st.info("Insufficient regime changes for event study analysis.")
        else:
            # Calculate pre/post windows for each trigger event
            pre_post_data = []
            
            for idx, row in trigger_events.iterrows():
                target_regime = row['Regime']
                trigger_idx = df_local.index.get_loc(idx)
                
                # Calculate percentage changes for pre windows
                pre_windows = {}
                for window in [30, 20, 10, 5, 1]:
                    start_idx = max(0, trigger_idx - window)
                    if start_idx < trigger_idx:
                        start_price = price_series.iloc[start_idx]
                        end_price = price_series.iloc[trigger_idx]
                        if start_price != 0:
                            pre_windows[f'Prior_{window}d'] = ((end_price / start_price) - 1) * 100
                        else:
                            pre_windows[f'Prior_{window}d'] = 0
                    else:
                        pre_windows[f'Prior_{window}d'] = 0
                
                # Calculate percentage changes for post windows
                post_windows = {}
                for window in [1, 5, 10, 20, 30]:
                    end_idx = min(len(price_series) - 1, trigger_idx + window)
                    if end_idx > trigger_idx:
                        start_price = price_series.iloc[trigger_idx]
                        end_price = price_series.iloc[end_idx]
                        if start_price != 0:
                            post_windows[f'Next_{window}d'] = ((end_price / start_price) - 1) * 100
                        else:
                            post_windows[f'Next_{window}d'] = 0
                    else:
                        post_windows[f'Next_{window}d'] = 0
                
                pre_post_data.append({
                    'Regime': target_regime,
                    **pre_windows,
                    **post_windows
                })
            
            # Create event study dataframe
            event_df = pd.DataFrame(pre_post_data)
            
            # Group by regime and calculate means
            event_summary = event_df.groupby('Regime').mean()
            event_summary = event_summary.reindex(regimes_order).fillna(0)
            
            # Split into pre and post tables
            pre_cols = ['Prior_30d', 'Prior_20d', 'Prior_10d', 'Prior_5d', 'Prior_1d']
            post_cols = ['Next_1d', 'Next_5d', 'Next_10d', 'Next_20d', 'Next_30d']
            
            pre_table = event_summary[pre_cols].copy()
            pre_table.columns = ['Prior 30d', 'Prior 20d', 'Prior 10d', 'Prior 5d', 'Prior 1d']
            
            post_table = event_summary[post_cols].copy()
            post_table.columns = ['Next 1d', 'Next 5d', 'Next 10d', 'Next 20d', 'Next 30d']
            
            # Format as percentages
            pre_table = pre_table.applymap(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
            post_table = post_table.applymap(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
            
            # Display in two columns
            col_pre, col_post = st.columns(2)
            
            with col_pre:
                st.markdown("**📊 Pre-Regime Context (The Lead-Up)**")
                st.caption("*Price changes BEFORE entering each regime*")
                st.dataframe(pre_table, use_container_width=True)
            
            with col_post:
                st.markdown("**📈 Post-Regime Outcome (The Aftermath)**")
                st.caption("*Price changes AFTER entering each regime*")
                st.dataframe(post_table, use_container_width=True)
        
        # === CRASH FORENSICS ===
        st.markdown("---")
        st.markdown("### 📉 Forensic Analysis: Anatomy of a Crash")
        st.caption("*Analyze price movements before and after entering CRITICAL regime*")
        
        # Identify crash events (regime switched TO CRITICAL)
        df_local['Prev_Regime'] = df_local['Regime'].shift(1)
        df_local['Is_Crash'] = (df_local['Regime'] == 'CRITICAL') & (df_local['Prev_Regime'] != 'CRITICAL')
        crash_events = df_local[df_local['Is_Crash']].index.tolist()
        
        if len(crash_events) < 2:
            st.info("Insufficient CRITICAL regime events for crash forensics (need at least 2).")
        else:
            # Extract lead-up and aftermath for each crash
            crash_data = []
            lead_up_regimes = {-10: [], -8: [], -5: [], -3: [], -1: []}
            
            for crash_date in crash_events:
                crash_idx = df_local.index.get_loc(crash_date)
                
                # Extract 10 days before and 10 days after
                start_idx = max(0, crash_idx - 10)
                end_idx = min(len(df_local) - 1, crash_idx + 10)
                
                if start_idx < crash_idx:
                    # Get price at start (for normalization)
                    start_price = price_series.iloc[start_idx]
                    
                    if start_price != 0:
                        # Extract prices and regimes for this crash event
                        for i in range(start_idx, end_idx + 1):
                            days_from_crash = i - crash_idx
                            price_norm = ((price_series.iloc[i] / start_price) - 1) * 100
                            regime = df_local['Regime'].iloc[i]
                            
                            crash_data.append({
                                'Days_From_Crash': days_from_crash,
                                'Price_Change_Pct': price_norm,
                                'Regime': regime
                            })
                            
                            # Collect regime at specific lead-up points
                            if days_from_crash in lead_up_regimes:
                                lead_up_regimes[days_from_crash].append(regime)
            
            if crash_data:
                crash_df = pd.DataFrame(crash_data)
                
                # Chart A: Warning Countdown (Stacked Bar)
                lead_up_points = [-10, -8, -5, -3, -1]
                regime_dist = []
                
                for day in lead_up_points:
                    regimes_at_day = lead_up_regimes[day]
                    total = len(regimes_at_day) if regimes_at_day else 1
                    dist = {}
                    for reg in regimes_order:
                        count = regimes_at_day.count(reg)
                        dist[reg] = (count / total) * 100
                    regime_dist.append(dist)
                
                fig_warning = go.Figure()
                for reg in regimes_order:
                    values = [regime_dist[i][reg] for i in range(len(lead_up_points))]
                    fig_warning.add_trace(go.Bar(
                        x=[f"{abs(d)}d before" for d in lead_up_points],
                        y=values,
                        name=reg,
                        marker_color=colors.get(reg, '#888'),
                        hovertemplate=f"{reg}: %{{y:.1f}}%<extra></extra>"
                    ))
                
                fig_warning.update_layout(
                    title="The Warning Countdown: Regime Distribution Before Crash",
                    barmode='stack',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Merriweather, serif", color=text_color, size=11),
                    margin=dict(l=40, r=20, t=50, b=40),
                    height=350,
                    yaxis_title="% of Occurrences",
                    xaxis_title="Time Before Crash"
                )
                fig_warning.update_yaxes(gridcolor='#E6E1D3', gridwidth=0.5)
                
                # Chart B: Average Crash Trajectory (Line)
                avg_trajectory = crash_df.groupby('Days_From_Crash')['Price_Change_Pct'].mean().reset_index()
                avg_trajectory = avg_trajectory.sort_values('Days_From_Crash')
                
                fig_trajectory = go.Figure()
                fig_trajectory.add_trace(go.Scatter(
                    x=avg_trajectory['Days_From_Crash'],
                    y=avg_trajectory['Price_Change_Pct'],
                    mode='lines+markers',
                    name='Average Price Path',
                    line=dict(color='#2C3E50', width=2),
                    marker=dict(size=6, color='#2C3E50')
                ))
                
                # Add vertical line at x=0 (signal trigger)
                fig_trajectory.add_vline(
                    x=0,
                    line=dict(color='#C0392B', width=2, dash='dash'),
                    annotation_text="Signal Trigger",
                    annotation_position="top"
                )
                
                fig_trajectory.update_layout(
                    title="Average Crash Trajectory: Price Path Around Signal",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Merriweather, serif", color=text_color, size=11),
                    margin=dict(l=40, r=20, t=50, b=40),
                    height=350,
                    yaxis_title="Price Change (%)",
                    xaxis_title="Days Relative to Crash Signal (0 = Trigger)"
                )
                fig_trajectory.update_xaxes(gridcolor='#E6E1D3', gridwidth=0.5, zeroline=True, zerolinecolor='#333', zerolinewidth=1)
                fig_trajectory.update_yaxes(gridcolor='#E6E1D3', gridwidth=0.5, zeroline=True, zerolinecolor='#333', zerolinewidth=1)
                
                # Display charts
                col_warn, col_traj = st.columns(2)
                with col_warn:
                    st.plotly_chart(fig_warning, use_container_width=True)
                with col_traj:
                    st.plotly_chart(fig_trajectory, use_container_width=True)
            else:
                st.info("No crash event data available for forensic analysis.")


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="TECTONIQ - Market Analysis Platform",
    page_icon="assets/logo-soc.png",
    layout="wide",
    initial_sidebar_state="collapsed"  # No sidebar used - user menu in header
)

# =============================================================================
# CONSTANTS
# =============================================================================
ACCESS_CODE = "BETA2025"
DEFAULT_SMA_WINDOW = 200
DEFAULT_VOL_WINDOW = 30
DEFAULT_HYSTERESIS = 0.0
MIN_DATA_POINTS = 200

# Precious metals excluded from main risk scan - they act as hedges (inverse correlation)
# and distort market risk scoring. Available separately in "Hedge Assets" category.
PRECIOUS_METALS = {'GC=F', 'SI=F', 'PL=F', 'PA=F', 'GLD', 'SLV'}

# Popular tickers for quick suggestions
POPULAR_TICKERS = {
    "US Tech": ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META'],
    "Crypto": ['BTC-USD', 'ETH-USD', 'SOL-USD'],
    "ETFs": ['SPY', 'QQQ', 'IWM', 'VTI'],
}

TICKER_NAME_FIXES = {
    "SIEMENS                    N": "Siemens", "Allianz                    v": "Allianz",
    "DEUTSCHE TELEKOM           N": "Deutsche Telekom", "Airbus                     A": "Airbus",
    "BAYERISCHE MOTOREN WERKE   S": "BMW", "VOLKSWAGEN                 V": "Volkswagen",
    "BASF                       N": "BASF", "MUENCHENER RUECKVERS.-GES. N": "Munich Re",
    "SAP                       ": "SAP"
}

SPECIAL_TICKER_NAMES = {"^GDAXI": "DAX 40 Index"}

# =============================================================================
# STYLING & THEME
# =============================================================================

def get_theme_css(is_dark: bool) -> str:
    """
    Generate comprehensive CSS styles for theme (dark/light mode).
    
    Handles styling for: app background, text colors, inputs, tables,
    buttons, cards, asset list items, and footer elements.
    """
    c = {
        "bg": "#0E1117" if is_dark else "#FFFFFF",
        "bg2": "#262730" if is_dark else "#F0F2F6",
        "card": "#1E1E1E" if is_dark else "#F8F9FA",
        "border": "#333" if is_dark else "#DEE2E6",
        "text": "#FAFAFA" if is_dark else "#212529",
        "muted": "#888" if is_dark else "#6C757D",
        "input": "#262730" if is_dark else "#FFFFFF"
    }
    
    return f"""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: {c['bg']} !important;
    }}
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stMarkdown, .stMarkdown p {{
        color: {c['text']} !important;
    }}
    .stTextInput input, .stTextArea textarea, [data-baseweb="select"], [data-baseweb="select"] > div {{
        background-color: {c['input']} !important;
        color: {c['text']} !important;
        border-color: {c['border']} !important;
    }}
    .streamlit-expanderHeader {{ 
        background-color: {c['card']} !important; 
        color: {c['text']} !important;
    }}
    .streamlit-expanderHeader p, .streamlit-expanderHeader span,
    .streamlit-expanderHeader svg {{ 
        color: {c['text']} !important; 
        fill: {c['text']} !important;
    }}
    .streamlit-expanderContent {{ 
        background-color: {c['bg2']} !important; 
        color: {c['text']} !important;
    }}
    .streamlit-expanderContent p, .streamlit-expanderContent span,
    .streamlit-expanderContent label, .streamlit-expanderContent div {{
        color: {c['text']} !important;
    }}
    [data-testid="stExpander"] {{
        background-color: {c['card']} !important;
        border-color: {c['border']} !important;
    }}
    [data-testid="stExpander"] details {{
        background-color: {c['card']} !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {c['text']} !important;
    }}
    .stDataFrame, [data-testid="stDataFrame"], .stDataFrame div, .stDataFrame table,
    .stDataFrame th, .stDataFrame td, [data-testid="glideDataEditor"], .dvn-scroller {{
        background-color: {c['card']} !important;
        color: {c['text']} !important;
    }}
    .stDataFrame th {{ background-color: {c['bg2']} !important; }}
    .stButton > button,
    [data-testid="baseButton-secondary"],
    [data-testid="stBaseButton-secondary"],
    button[kind="secondary"] {{
        background-color: {c['card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        font-weight: 600;
        border-radius: 8px;
    }}
    .stButton > button:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover {{ 
        background-color: {c['bg2']} !important; 
        color: {c['text']} !important;
        border-color: #667eea !important;
    }}
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"],
    [data-testid="stBaseButton-primary"],
    button[kind="primary"] {{
        background-color: #667eea !important;
        color: white !important;
        border-color: #667eea !important;
    }}
    .stButton > button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover {{
        background-color: #5568d9 !important;
    }}
    /* Ensure all button text is visible */
    .stButton button p,
    .stButton button span,
    [data-testid="baseButton-secondary"] p,
    [data-testid="baseButton-secondary"] span {{
        color: {c['text']} !important;
    }}
    .stRadio label {{ color: {c['text']} !important; }}
    .stRadio [role="radiogroup"] label {{ background-color: {c['card']} !important; border-color: {c['border']} !important; }}
    [data-testid="stMetricValue"] {{ color: {c['text']} !important; }}
    [data-testid="stMetricLabel"] {{ color: {c['muted']} !important; }}
    [data-testid="stSidebar"] {{ display: none; }}
    .stDeployButton {{ visibility: hidden; }}
    .block-container {{ padding-top: 2rem; max-width: 1400px; margin: 0 auto; }}
    hr {{ border-color: {c['border']} !important; }}
    .app-header {{
        display: flex; align-items: center; gap: 1rem;
        padding: 0.75rem 0; border-bottom: 2px solid {c['border']}; margin-bottom: 1rem;
    }}
    .logo {{ width: 45px; height: 45px; background: linear-gradient(135deg, #667eea, #764ba2);
             border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; }}
    .app-title {{ font-size: 1.6rem; font-weight: 700;
                  background: linear-gradient(90deg, #667eea, #764ba2);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }}
    .app-subtitle {{ font-size: 0.8rem; color: {c['muted']} !important; margin: 0; }}
    .asset-item {{
        padding: 0.6rem 0.8rem; border-radius: 6px; margin-bottom: 4px;
        cursor: pointer; transition: all 0.15s ease;
        border: 1px solid transparent;
    }}
    .asset-item:hover {{ background-color: {c['bg2']}; }}
    .asset-item.selected {{ background-color: {c['bg2']}; border-color: #667eea; }}
    .asset-symbol {{ font-weight: 600; font-size: 0.95rem; }}
    .asset-price {{ color: {c['muted']}; font-size: 0.85rem; }}
    .asset-signal {{ font-size: 0.8rem; }}
    .detail-header {{ padding: 1rem; background: {c['card']}; border-radius: 8px; margin-bottom: 1rem; }}
    .signal-badge {{
        display: inline-block; padding: 0.4rem 0.8rem; border-radius: 6px;
        font-weight: 600; font-size: 0.9rem;
    }}
    .footer {{ border-top: 1px solid {c['border']}; padding-top: 0.75rem; margin-top: 1.5rem; }}
    .footer-item {{ display: inline-block; margin-right: 1.5rem; font-size: 0.8rem; color: {c['muted']}; }}
</style>
"""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_name(name: str) -> str:
    """
    Clean and normalize ticker names from Yahoo Finance.
    
    Handles German stock suffixes (SE, AG), fixes common formatting issues,
    and applies hardcoded fixes for known problematic names (DAX stocks).
    """
    name = name.replace(" SE", "").replace(" AG", "").strip()
    if name in TICKER_NAME_FIXES:
        return TICKER_NAME_FIXES[name]
    name = " ".join(name.split())
    return name[:-2] if len(name) > 2 and name[-2] == " " else name


def get_signal_color(signal: str) -> str:
    """
    Get display color for 5-tier regime classification.
    
    Mapping:
        STABLE → Green (#00FF00)
        ACTIVE → Yellow (#FFCC00)
        HIGH_ENERGY → Orange (#FF6600)
        CRITICAL → Red (#FF0000)
        DORMANT → Grey (#888888)
    """
    signal_upper = signal.upper()
    if "STABLE" in signal_upper:
        return "#00FF00"
    if "CRITICAL" in signal_upper:
        return "#FF0000"
    if "HIGH_ENERGY" in signal_upper or "HIGH ENERGY" in signal_upper:
        return "#FF6600"
    if "ACTIVE" in signal_upper:
        return "#FFCC00"
    if "DORMANT" in signal_upper:
        return "#888888"
    return "#888888"


def get_signal_bg(signal: str) -> str:
    """Get semi-transparent background color for regime badge display."""
    signal_upper = signal.upper()
    if "STABLE" in signal_upper:
        return "rgba(0, 255, 0, 0.15)"
    if "CRITICAL" in signal_upper:
        return "rgba(255, 0, 0, 0.15)"
    if "HIGH_ENERGY" in signal_upper or "HIGH ENERGY" in signal_upper:
        return "rgba(255, 102, 0, 0.15)"
    if "ACTIVE" in signal_upper:
        return "rgba(255, 204, 0, 0.15)"
    if "DORMANT" in signal_upper:
        return "rgba(136, 136, 136, 0.2)"
    return "rgba(136, 136, 136, 0.15)"


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def run_analysis(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Run SOC analysis on multiple tickers with progress indicator.
    
    For each ticker: fetches data, calculates metrics, determines market
    phase (5-tier classification), and returns analysis results.
    
    Includes robust API error handling with user-friendly messages.
    
    Returns:
        List of dictionaries containing symbol, price, signal, trend,
        criticality_score, and other phase metrics.
    """
    fetcher = DataFetcher(cache_enabled=True)
    results = []
    failed_tickers = []
    api_error_count = 0
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, symbol in enumerate(tickers):
        status.caption(f"Analyzing {symbol}...")
        try:
            df = fetcher.fetch_data(symbol)
            info = fetcher.fetch_info(symbol)
            if not df.empty and len(df) > MIN_DATA_POINTS:
                analyzer = SOCAnalyzer(df, symbol, info, DEFAULT_SMA_WINDOW, DEFAULT_VOL_WINDOW, DEFAULT_HYSTERESIS)
                phase = analyzer.get_market_phase()
                phase['info'] = info
                phase['name'] = clean_name(info.get('name', symbol))
                results.append(phase)
            else:
                failed_tickers.append(symbol)
                api_error_count += 1
        except ConnectionError:
            failed_tickers.append(symbol)
            api_error_count += 1
        except TimeoutError:
            failed_tickers.append(symbol)
            api_error_count += 1
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or 'too many requests' in error_msg:
                api_error_count += 1
            elif 'connection' in error_msg or 'timeout' in error_msg or 'network' in error_msg:
                api_error_count += 1
            failed_tickers.append(symbol)
        
        progress.progress((i + 1) / len(tickers))
    
    status.empty()
    progress.empty()
    
    # Show API error warning if significant failures
    if api_error_count >= len(tickers) * 0.5:  # More than 50% failed
        st.error("""
        ⚠️ **Data Provider Unavailable**
        
        The market data API (Yahoo Finance) appears to be experiencing issues.
        This could be due to:
        - Rate limiting (too many requests)
        - Temporary service outage
        - Network connectivity issues
        
        **Please try again in 5-10 minutes.**
        
        If the problem persists, check [Yahoo Finance Status](https://finance.yahoo.com).
        """)
    elif failed_tickers:
        st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers[:5])}{'...' if len(failed_tickers) > 5 else ''}")
    
    return results


# =============================================================================
# TICKER SEARCH & VALIDATION FUNCTIONS
# =============================================================================

def search_ticker(query: str) -> List[Dict[str, Any]]:
    """
    Search for tickers by company name or symbol using Yahoo Finance.
    
    Args:
        query: Search term (company name or ticker symbol)
    
    Returns:
        List of matching results with ticker, name, type, exchange
    """
    if not query or len(query) < 2:
        return []
    
    try:
        # Use Yahoo Finance search API
        url = f"https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            'q': query,
            'quotesCount': 8,
            'newsCount': 0,
            'enableFuzzyQuery': True,
            'quotesQueryId': 'tss_match_phrase_query'
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for quote in data.get('quotes', []):
            # Filter for stocks, ETFs, and crypto only
            quote_type = quote.get('quoteType', '').upper()
            if quote_type in ['EQUITY', 'ETF', 'CRYPTOCURRENCY', 'INDEX', 'MUTUALFUND']:
                results.append({
                    'ticker': quote.get('symbol', ''),
                    'name': quote.get('shortname') or quote.get('longname') or quote.get('symbol', ''),
                    'type': quote_type,
                    'exchange': quote.get('exchange', '')
                })
        
        return results
        
    except requests.exceptions.RequestException:
        return []
    except Exception:
        return []


def validate_ticker(ticker: str) -> Dict[str, Any]:
    """
    Validate a ticker symbol using yfinance.
    
    Args:
        ticker: Stock/crypto ticker symbol (e.g., 'AAPL', 'BTC-USD')
    
    Returns:
        Dict with 'valid' (bool), 'name' (str), 'error' (str if invalid)
    """
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return {'valid': False, 'error': 'Empty ticker'}
        
        stock = yf.Ticker(ticker)
        
        # Try to get recent price history first (most reliable check)
        hist = stock.history(period='5d')
        
        if hist.empty:
            return {'valid': False, 'error': f'Ticker "{ticker}" not found'}
        
        # Get latest price from history
        latest_price = hist['Close'].iloc[-1] if not hist.empty else 0
        
        # Try to get info for name (but don't fail if unavailable)
        name = ticker
        try:
            info = stock.info
            if info:
                # Try multiple name fields
                name = (info.get('shortName') or 
                       info.get('longName') or 
                       info.get('name') or 
                       ticker)
                # Try multiple price fields if history price is 0
                if latest_price == 0:
                    latest_price = (info.get('regularMarketPrice') or 
                                  info.get('previousClose') or 
                                  info.get('currentPrice') or 
                                  0)
        except:
            # If info fails, just use ticker as name
            pass
        
        return {
            'valid': True,
            'ticker': ticker,
            'name': name,
            'price': latest_price
        }
        
    except requests.exceptions.RequestException:
        return {'valid': False, 'error': 'API not responding. Please try again in 5 minutes.'}
    except Exception as e:
        error_msg = str(e).lower()
        if 'connection' in error_msg or 'timeout' in error_msg or 'network' in error_msg:
            return {'valid': False, 'error': 'API not responding. Please try again in 5 minutes.'}
        return {'valid': False, 'error': f'Ticker "{ticker}" not found'}


# =============================================================================
# DETAIL PANEL UI COMPONENTS
# =============================================================================
# NOTE: Detail panel UI moved to ui_detail.py
# Import: from ui_detail import render_detail_panel, render_regime_persistence_chart, render_current_regime_outlook



# =============================================================================
# INVESTMENT SIMULATION UI
# =============================================================================

# =============================================================================
# AUTHENTICATION & NAVIGATION UI
# =============================================================================
# NOTE: Auth & navigation UI moved to ui_auth.py
# Import: from ui_auth import render_disclaimer, check_auth, login_page, render_sticky_cockpit_header, render_education_landing


# =============================================================================
# LEGAL PAGE DIALOGS
# =============================================================================

@st.dialog("⚖️ Legal Disclaimer", width="large")
def show_disclaimer_dialog():
    """Show disclaimer in a modal dialog."""
    st.markdown("""
    This application is provided for educational and informational purposes only.
    Nothing on this platform constitutes financial, investment, or trading advice.
    
    **No Investment Recommendations:**
    - We do not recommend buying, selling, or holding any financial instruments
    - All analysis is purely statistical observation
    - Past performance is not indicative of future results
    
    **Limitation of Liability:**
    - The creators shall not be liable for any damages arising from use
    - Users accept full responsibility for their investment decisions
    
    **Independent Verification Required:**
    - Consult with qualified financial advisors before making decisions
    - Conduct your own research and due diligence
    
    ---
    
    *This disclaimer is governed by applicable laws. If any provision is found unenforceable,
    the remaining provisions shall continue in full force and effect.*
    """)
    
    if st.button("Close", key="close_disclaimer", use_container_width=True):
        st.rerun()


@st.dialog("🔒 Data Protection Policy", width="large")
def show_data_protection_dialog():
    """Show data protection policy in a modal dialog."""
    st.markdown("""
    **Data Controller:** TECTONIQ Platform
    
    **Data We Collect:**
    - Email address (for authentication)
    - Portfolio preferences (ticker symbols you save)
    - Usage analytics (anonymous)
    
    **How We Use Your Data:**
    - To provide authentication services
    - To save your portfolio preferences
    - To improve the application
    
    **Data Storage:**
    - Stored securely via Supabase (EU servers)
    - Encrypted in transit and at rest
    - Not shared with third parties
    
    **Your Rights:**
    - Right to access your data
    - Right to delete your account
    - Right to data portability
    
    **GDPR Compliance:**
    - We comply with EU data protection regulations
    - Data processing is limited to stated purposes
    - You can request data deletion at any time
    
    **Contact:** For data protection inquiries, email privacy@tectoniq.app
    """)
    
    if st.button("Close", key="close_data_protection", use_container_width=True):
        st.rerun()


@st.dialog("📄 Imprint / Legal Notice", width="large")
def show_imprint_dialog():
    """Show imprint in a modal dialog."""
    st.markdown("""
    **Service Provider:**  
    TECTONIQ Platform  
    [Your Address]  
    [City, Postal Code]  
    [Country]
    
    **Contact:**  
    Email: info@tectoniq.app  
    Web: tectoniq.app
    
    **Responsible for Content:**  
    [Your Name / Company Name]
    
    **Disclaimer:**  
    This platform provides educational content only. We assume no liability for the 
    accuracy, completeness, or timeliness of the information provided.
    
    **Copyright:**  
    © 2025 TECTONIQ. All rights reserved. Unauthorized reproduction or distribution 
    of this application or its content is prohibited.
    
    **Third-Party Data:**  
    Market data provided by Yahoo Finance. We do not control or guarantee the 
    accuracy of third-party data sources.
    
    **Jurisdiction:**  
    This imprint is provided in accordance with applicable laws and regulations.
    """)
    
    if st.button("Close", key="close_imprint", use_container_width=True):
        st.rerun()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application entry point - Multi-User SaaS Edition.
    
    Flow:
        1. Show legal disclaimer (must accept to continue)
        2. Check user authentication (Supabase)
        3. Initialize session state
        4. Apply theme CSS
        5. Render sidebar with logout + user info
        6. Render sticky cockpit header (always visible)
        7. Main content area:
           - No asset: Education landing + quick picks
           - Asset selected: Deep Dive or Simulation tabs (tier-gated)
    """
    # Session state initialization
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state.disclaimer_accepted = False
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False  # Light mode default
    if 'selected_asset' not in st.session_state:
        st.session_state.selected_asset = 0
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "deep_dive"
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None
    
    # Apply Scientific Heritage CSS theme FIRST (before any page)
    st.markdown(get_scientific_heritage_css(), unsafe_allow_html=True)
    
    # Legal disclaimer gate (must accept before anything else)
    if not st.session_state.disclaimer_accepted:
        render_disclaimer()
        return
    
    # === AUTHENTICATION GATE (THE GATEKEEPER) ===
    # Check if user is authenticated via Supabase
    if not is_authenticated():
        render_auth_page()
        return
    
    # === STICKY COCKPIT HEADER (with User Menu) ===
    render_sticky_cockpit_header(validate_ticker, search_ticker, run_analysis)
    
    # === PORTFOLIO VIEW (if toggled on) ===
    if st.session_state.get('show_portfolio', False):
        with st.container(border=True):
            col_header, col_close = st.columns([4, 1])
            with col_header:
                st.markdown("### 📁 My Portfolio")
            with col_close:
                if st.button("Close portfolio window", key="close_portfolio", use_container_width=True):
                    st.session_state.show_portfolio = False
                    st.rerun()
            
            user_id = get_current_user_id()
            user_tier = st.session_state.get('tier', 'free')
            
            if user_id:
                portfolio = get_user_portfolio(user_id)
                if portfolio:
                    st.caption(f"**{len(portfolio)}** assets tracked")
                    st.markdown("---")
                    
                    # Fetch full analysis for all portfolio assets (includes crash_warning)
                    with st.spinner("Loading portfolio data..."):
                        import time
                        portfolio_analysis = []
                        fetcher = DataFetcher(cache_enabled=True)
                        failed_tickers = []
                        
                        for i, ticker in enumerate(portfolio):
                            try:
                                # Add small delay between requests to avoid rate limiting
                                if i > 0:
                                    time.sleep(0.5)  # 500ms delay between tickers
                                
                                df = fetcher.fetch_data(ticker)
                                info = fetcher.fetch_info(ticker)
                                if not df.empty and len(df) > MIN_DATA_POINTS:
                                    analyzer = SOCAnalyzer(df, ticker, info, DEFAULT_SMA_WINDOW, DEFAULT_VOL_WINDOW, DEFAULT_HYSTERESIS)
                                    phase = analyzer.get_market_phase()
                                    
                                    # Get full analysis including crash_warning for accurate stress level
                                    try:
                                        full_analysis = analyzer.get_full_analysis()
                                        crash_warning = full_analysis.get('crash_warning', {})
                                        
                                        # Ensure crash_warning has a score
                                        if crash_warning and 'score' in crash_warning:
                                            phase['crash_warning'] = crash_warning
                                        else:
                                            print(f"Warning: {ticker} crash_warning missing score, recalculating...")
                                            # Force recalculation if missing
                                            phase['crash_warning'] = full_analysis.get('crash_warning', {'score': 0})
                                    except Exception as analysis_error:
                                        # Fallback: calculate basic stress if full analysis completely fails
                                        print(f"Full analysis failed for {ticker}: {str(analysis_error)}")
                                        phase['crash_warning'] = {'score': 0}
                                    
                                    phase['name'] = clean_name(info.get('name', ticker))
                                    portfolio_analysis.append(phase)
                                else:
                                    failed_tickers.append(ticker)
                            except Exception as e:
                                failed_tickers.append(ticker)
                                print(f"Error loading {ticker}: {str(e)}")  # Debug log
                        
                        # Show warning/error based on results
                        if failed_tickers and portfolio_analysis:
                            # Some tickers failed, but others loaded successfully
                            st.warning(f"⚠️ Could not load data for: {', '.join(failed_tickers)}")
                        elif failed_tickers and not portfolio_analysis:
                            # All tickers failed
                            st.error(f"""
                            ❌ **Could not load portfolio data**
                            
                            Failed to fetch: {', '.join(failed_tickers)}
                            
                            **Possible reasons:**
                            - Yahoo Finance API rate limiting (try again in 1-2 minutes)
                            - Network connectivity issues
                            - Invalid ticker symbols
                            
                            💡 **Tip:** Try removing and re-adding the assets, or search for them individually first.
                            """)
                    
                    if portfolio_analysis:
                        # Create table data
                        table_data = []
                        for result in portfolio_analysis:
                            table_data.append({
                                "Ticker": result['symbol'],
                                "Asset Name": result.get('name', result['symbol']),
                                "Criticality": int(result.get('criticality_score', 0)),
                                "Regime": result.get('signal', 'Unknown'),
                                "_result": result  # Store full result for actions
                            })
                        
                        # Sort by criticality (highest first)
                        table_data.sort(key=lambda x: x['Criticality'], reverse=True)
                        
                        # Display table
                        for i, row in enumerate(table_data):
                            # Color code criticality
                            crit = row['Criticality']
                            if crit > 80:
                                crit_color = "#FF4040"
                            elif crit > 60:
                                crit_color = "#FF6600"
                            else:
                                crit_color = "#00C864"
                            
                            # Get regime emoji
                            regime_text = row['Regime']
                            regime_emoji = regime_text.split()[0] if regime_text else "⚪"
                            
                            # Row container
                            with st.container():
                                col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 1, 1])
                                
                                with col1:
                                    st.markdown(f"**{row['Ticker']}**")
                                
                                with col2:
                                    st.markdown(f"{row['Asset Name']}")
                                
                                with col3:
                                    st.markdown(f"{regime_emoji} <span style='color: {crit_color}; font-weight: 600;'>Criticality: {crit}</span>", unsafe_allow_html=True)
                                
                                with col4:
                                    if st.button("→ Deep Dive", key=f"deepdive_{row['Ticker']}", use_container_width=True):
                                        # Load this asset
                                        st.session_state.current_ticker = row['Ticker']
                                        st.session_state.scan_results = [row['_result']]
                                        st.session_state.selected_asset = 0
                                        st.session_state.analysis_mode = "deep_dive"
                                        st.session_state.show_portfolio = False  # Close portfolio
                                        st.rerun()
                                
                                with col5:
                                    if st.button("🗑️", key=f"remove_{row['Ticker']}", help="Remove from portfolio", use_container_width=True):
                                        from auth_manager import remove_asset_from_portfolio
                                        success, error = remove_asset_from_portfolio(user_id, row['Ticker'])
                                        if success:
                                            st.rerun()
                                        else:
                                            st.error(error)
                                
                                st.markdown("<hr style='margin: 8px 0; opacity: 0.2;'>", unsafe_allow_html=True)
            else:
                st.warning("Please log in to view your portfolio.")
        
        st.markdown("---")
            
    # === TICKER SUGGESTIONS (if user searched by company name) ===
    if 'ticker_suggestions' in st.session_state and st.session_state.ticker_suggestions:
        # Centered info box
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.info("🔍 Not a ticker symbol. Did you mean one of these?")
        
        st.markdown("#### Select a ticker:")
        suggestions = st.session_state.ticker_suggestions[:6]  # Max 6 suggestions
        
        num_cols = min(3, len(suggestions))
        cols = st.columns(num_cols)
        
        for i, suggestion in enumerate(suggestions):
            col_idx = i % num_cols
            ticker = suggestion.get('ticker', '') or suggestion.get('symbol', '')  # Handle both keys
            
            # Skip empty tickers
            if not ticker:
                continue
                
            name = suggestion.get('name', ticker)[:25]
            exchange = suggestion.get('exchange', '')
            
            with cols[col_idx]:
                btn_label = f"{ticker}\n{name}"
                if exchange:
                    btn_label += f"\n({exchange})"
                
                if st.button(btn_label, key=f"suggest_{ticker}_{i}", use_container_width=True):
                    # Clear suggestions and analyze this ticker
                    st.session_state.ticker_suggestions = []
                    st.session_state.current_ticker = ticker
                    
                    # Run analysis
                    with st.spinner(f"Analyzing {ticker}..."):
                        try:
                            results = run_analysis([ticker])
                            if results and len(results) > 0:
                                st.session_state.scan_results = results
                                st.session_state.selected_asset = 0
                                st.session_state.analysis_mode = "deep_dive"
                                st.rerun()
                            else:
                                st.error(f"No data available for {ticker}. Try a different exchange variant.")
                        except Exception as e:
                            st.error(f"Error analyzing {ticker}: {str(e)}")
        
        # Clear button
        if st.button("✕ Clear suggestions", key="clear_suggestions"):
            st.session_state.ticker_suggestions = []
            st.rerun()
        
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # === MAIN CONTENT AREA (Dynamic) ===
    if 'scan_results' not in st.session_state or not st.session_state.scan_results:
        # CONDITION A: No Asset Selected - Show Education Landing
        render_education_landing(run_analysis)
    else:
        # CONDITION B: Asset Selected - Show Analysis
        results = st.session_state.scan_results
        
        # === ANALYSIS MODE TABS (all in one row) ===
        col_spacer1, col_tab1, col_tab2, col_spacer2 = st.columns([1, 2, 2, 1])
        
        with col_tab1:
            if st.button(
                "📊 Asset Deep Dive",
                key="btn_deep_dive",
                use_container_width=True
            ):
                st.session_state.analysis_mode = "deep_dive"
                st.rerun()
        
        with col_tab2:
            if st.button(
                "🎯 Portfolio Simulation",
                key="btn_simulation",
                use_container_width=True
            ):
                st.session_state.analysis_mode = "simulation"
                st.rerun()
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        # Active asset card removed - hero card is shown in deep dive section instead
        
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        
        # === RENDER SELECTED ANALYSIS ===
        if st.session_state.analysis_mode == "deep_dive":
            # Deep Dive Analysis
            if results:
                selected = results[st.session_state.selected_asset]
                
                # === SPECIMEN HERO CARD (Condensed) ===
                symbol = selected.get('symbol', '')
                name = selected.get('name', symbol)
                full_name = selected.get('info', {}).get('longName', name)
                price = selected.get('price', 0.0)
                change = selected.get('price_change_1d', selected.get('change_pct', 0.0))
                trend = selected.get('trend', 'Unknown')
                criticality = int(selected.get('criticality_score', 0))
                signal = selected.get('signal', 'Unknown')
                vol_pct = selected.get('vol_percentile', 0)
                
                # Simple regime strip color mapping
                sig_lower = signal.lower()
                if "critical" in sig_lower:
                    regime_color = "#C0392B"; regime_label = "CRITICAL REGIME"
                elif "high" in sig_lower:
                    regime_color = "#D35400"; regime_label = "HIGH ENERGY REGIME"
                elif "active" in sig_lower:
                    regime_color = "#F1C40F"; regime_label = "ACTIVE REGIME"
                elif "stable" in sig_lower:
                    regime_color = "#27AE60"; regime_label = "STABLE REGIME"
                else:
                    regime_color = "#95A5A6"; regime_label = signal.upper()
                
                # Criticality badge color
                if criticality > 80:
                    crit_color = "#C0392B"
                elif criticality > 60:
                    crit_color = "#D35400"
                elif criticality > 40:
                    crit_color = "#F1C40F"
                else:
                    crit_color = "#27AE60"
                
                # Basic persistence and win rate (with safer fallbacks)
                persistence = (
                    selected.get('current_streak_days')
                    or selected.get('current_streak')
                    or selected.get('streak')
                    or "N/A"
                )
                win_rate_raw = (
                    selected.get('win_rate')
                    or selected.get('probability')
                    or selected.get('historical_probability')
                )
                win_rate = f"{win_rate_raw:.0f}%" if isinstance(win_rate_raw, (int, float)) else "N/A"
                context_text = selected.get('context') or f"Regime: {regime_label}."
                # Normalize strings for formatting
                if isinstance(persistence, str):
                    persistence_display = persistence
                else:
                    persistence_display = f"{persistence}"
                win_rate_display = win_rate if isinstance(win_rate, str) else f"{win_rate}"
                
                # Center the card at 50% width
                col_left_card, col_center_card, col_right_card = st.columns([1, 2, 1])
                with col_center_card:
                    card_html = f"""
<div style="border:1px double #333; border-radius:8px; background:#FFFFFF; padding:16px 18px; margin-bottom:12px; box-shadow:2px 2px 6px rgba(0,0,0,0.05); font-family:'Merriweather', serif; width:100%;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
    <div>
      <div style="font-size:1.4rem; font-weight:700; color:#2C3E50;">{name}</div>
      <div style="font-size:1rem; color:#555;">{full_name}</div>
    </div>
    <div style="min-width:64px; text-align:center;">
      <div style="width:64px; height:64px; border-radius:50%; background:{crit_color}; color:#fff; display:flex; align-items:center; justify-content:center; font-size:1.4rem; font-weight:800; box-shadow:0 2px 6px rgba(0,0,0,0.15);">{criticality}</div>
      <div style="font-size:0.75rem; color:#555; margin-top:4px;">Criticality</div>
    </div>
  </div>

  <div style="margin:12px 0; padding:6px 10px; background:{regime_color}; color:#fff; font-weight:700; letter-spacing:0.5px; border-radius:4px; text-transform:uppercase;">{regime_label}</div>

  <div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:8px;">
    <div>
      <div style="font-size:2rem; font-weight:800; color:#2C3E50;">${price:,.2f}</div>
      <div style="font-size:0.95rem; color:{'#27AE60' if change >=0 else '#C0392B'};">{change:+.2f}%</div>
    </div>
  </div>

  <div style="margin-top:12px; padding:10px 12px; background:#F9F7F1; border:1px solid #D1C4E9; border-radius:6px; font-size:0.95rem; line-height:1.5; color:#333; font-family:'Merriweather', serif;">
    Persistence: {persistence_display} days. Historical Probability: {win_rate_display}.
    Context: {context_text}
  </div>

  <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; font-size:0.95rem; color:#333;">
    <div>Trend: <strong>{trend}</strong></div>
    <div>Volatility: <strong>{vol_pct:.0f}th %ile</strong></div>
  </div>
</div>
"""
                    st.markdown(card_html, unsafe_allow_html=True)
                
                # === SOC Chart (Plotly) ===
                is_dark = st.session_state.get('dark_mode', False)
                fetcher = DataFetcher(cache_enabled=True)
                df = fetcher.fetch_data(symbol)
                if not df.empty:
                    analyzer = SOCAnalyzer(df, symbol, selected.get('info'))
                    figs = analyzer.get_plotly_figures(dark_mode=is_dark)
                    st.plotly_chart(figs['chart3'], width="stretch")
                    
                    # Advanced analytics (visual-first)
                    render_advanced_analytics(df, is_dark=is_dark)
                else:
                    st.warning("No data available for this asset.")
        else:
            # Portfolio Simulation (unlimited for all users)
            st.markdown("### DCA Simulation")
            st.markdown("---")
            
            result_tickers = [r['symbol'] for r in results]
            render_dca_simulation(result_tickers)
    
    # === FOOTER WITH LEGAL LINKS ===
    st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; font-family: 'Merriweather', serif; font-size: 0.85rem; color: #666;">
        <p style="margin: 0 0 12px 0;">© 2025 TECTONIQ. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Legal page buttons (open as modal dialogs)
    col_spacer1, col1, col_sep1, col2, col_sep2, col3, col_spacer2 = st.columns([2, 1, 0.3, 1, 0.3, 1, 2])
    
    with col1:
        if st.button("Disclaimer", key="footer_disclaimer", use_container_width=True):
            show_disclaimer_dialog()
    
    with col_sep1:
        st.markdown("<p style='text-align: center; color: #BDC3C7; margin-top: 8px; font-size: 1.2rem;'>|</p>", unsafe_allow_html=True)
    
    with col2:
        if st.button("Data Protection", key="footer_data_protection", use_container_width=True):
            show_data_protection_dialog()
    
    with col_sep2:
        st.markdown("<p style='text-align: center; color: #BDC3C7; margin-top: 8px; font-size: 1.2rem;'>|</p>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Imprint", key="footer_imprint", use_container_width=True):
            show_imprint_dialog()


if __name__ == "__main__":
    main()
