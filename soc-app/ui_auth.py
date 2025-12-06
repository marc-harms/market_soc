"""
SOC Market Seismograph - Authentication & Navigation UI Components
===================================================================

Landing page, authentication, and sticky header components.

Contains:
- render_disclaimer(): Legal disclaimer page (must accept before using app)
- check_auth(): Authentication validation
- login_page(): Login UI
- render_sticky_cockpit_header(): Persistent search/status header
- render_education_landing(): Welcome page with quick start guide

Author: Market Analysis Team
Version: 8.0 (Modularized)
"""

from typing import Callable, Optional

import streamlit as st

from config import LEGAL_DISCLAIMER
from auth_manager import login, signup, logout, is_authenticated


def render_disclaimer() -> None:
    """
    Render legal disclaimer page that must be accepted before using the app.
    Uses session state to track acceptance.
    """
    st.markdown("""
    <style>
        .stApp { background-color: #0E1117; }
        .disclaimer-box {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #333;
            border-radius: 12px;
            padding: 24px;
            margin: 20px 0;
        }
    </style>
                    """, unsafe_allow_html=True)
                
    st.title("🔬 SOC Market Seismograph")
    st.caption("Self-Organized Criticality Analysis Tool")
    
    st.markdown("---")
    
    # Display disclaimer in scrollable container
    with st.container():
        st.markdown(LEGAL_DISCLAIMER)
    
    st.markdown("---")
    
    # Acceptance checkbox and button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        accepted = st.checkbox(
            "I have read, understood, and agree to the terms above",
            key="disclaimer_checkbox"
        )
        
        if st.button(
            "✅ I Understand & Accept",
            type="primary",
            disabled=not accepted,
            use_container_width=True
        ):
            st.session_state.disclaimer_accepted = True
            st.rerun()
        
        if not accepted:
            st.caption("⚠️ You must check the box above to continue")
    
    st.stop()


def render_auth_page() -> None:
    """
    Render modern login/signup page with tabs.
    
    This replaces the old simple access code system with full user authentication.
    Supports both login for existing users and signup for new users.
    """
    # Apply minimal styling for auth page
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .auth-container {
            background: white;
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            max-width: 450px;
            margin: 0 auto;
        }
        .auth-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .auth-header h1 {
            color: #667eea;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .auth-header p {
            color: #666;
            font-size: 0.95rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Center the auth form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="auth-header">
            <h1>🔬 SOC Seismograph</h1>
            <p>Self-Organized Criticality Market Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for Login / Sign Up
        tab1, tab2 = st.tabs(["🔑 Login", "✨ Sign Up"])
        
        # === LOGIN TAB ===
        with tab1:
            st.markdown("### Welcome Back")
            
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", placeholder="your@email.com", key="login_email")
                password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
                
                submit = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
                
                if submit:
                    if not email or not password:
                        st.error("Please enter both email and password")
                    else:
                        with st.spinner("Authenticating..."):
                            success, error, user_data = login(email, password)
                            
                            if success:
                                st.session_state.user = user_data
                                st.session_state.tier = user_data['tier']
                                st.session_state.authenticated = True
                                st.success(f"✅ Welcome back, {user_data['email']}!")
                                st.rerun()
                            else:
                                st.error(f"❌ {error}")
        
        # === SIGNUP TAB ===
        with tab2:
            st.markdown("### Create Your Account")
            st.caption("Start with our **Free tier** - no credit card required!")
            
            with st.form("signup_form", clear_on_submit=False):
                email = st.text_input("Email", placeholder="your@email.com", key="signup_email")
                password = st.text_input("Password", type="password", placeholder="Min. 6 characters", key="signup_password")
                password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="signup_password_confirm")
                
                agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
                
                submit = st.form_submit_button("✨ Create Account", use_container_width=True, type="primary")
                
                if submit:
                    # Validation
                    if not email or not password or not password_confirm:
                        st.error("Please fill in all fields")
                    elif password != password_confirm:
                        st.error("Passwords do not match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif not agree_terms:
                        st.error("Please agree to the Terms of Service")
                    else:
                        with st.spinner("Creating your account..."):
                            success, error, user_data = signup(email, password)
                            
                            if success:
                                st.session_state.user = user_data
                                st.session_state.tier = user_data['tier']
                                st.session_state.authenticated = True
                                st.success(f"🎉 Account created! Welcome, {user_data['email']}!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"❌ {error}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.85rem;">
            <p>🔒 Secure authentication powered by Supabase</p>
            <p>Questions? Contact <a href="mailto:support@socseismograph.com">support@socseismograph.com</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()


def render_sticky_cockpit_header(validate_ticker_func: Callable, search_ticker_func: Callable, run_analysis_func: Callable) -> None:
    """
    Render the persistent "Cockpit" header with search and status.
    
    Layout:
        Row 1: Logo | Search Field | Status Badge
        Row 2: Daily Summary Metrics (planned feature - placeholder for now)
    
    Args:
        validate_ticker_func: Function to validate ticker symbols
        search_ticker_func: Function to search for ticker by company name
        run_analysis_func: Function to run analysis on ticker(s)
    """
    is_dark = st.session_state.get('dark_mode', False)
    
    with st.container(border=True):
        # === ROW 1: Logo | Search | Status ===
        col_logo, col_search, col_status = st.columns([2, 4, 2])
        
        with col_logo:
            st.markdown('<h4 style="margin: 0;">📉 SOC Seismograph</h4>', unsafe_allow_html=True)
        
        with col_search:
            # Search with inline button
            search_col, btn_col = st.columns([4, 1])
            
            with search_col:
                search_query = st.text_input(
                    "Analyze Asset...",
                    placeholder="Type ticker (e.g., AAPL, BTC-USD)",
                    label_visibility="collapsed",
                    key="cockpit_search"
                )
            
            with btn_col:
                if st.button("🔍", key="search_btn", help="Analyze", use_container_width=True):
                    if search_query and len(search_query) > 0:
                        ticker_input = search_query.strip().upper()
                        
                        # Try to find ticker if user typed company name
                        with st.spinner(f"Searching for {ticker_input}..."):
                            try:
                                # First try direct ticker
                                validation = validate_ticker_func(ticker_input)
                                
                                if validation.get('valid'):
                                    # Valid ticker - analyze it
                                    results = run_analysis_func([ticker_input])
                                    if results and len(results) > 0:
                                        st.session_state.current_ticker = ticker_input
                                        st.session_state.scan_results = results
                                        st.session_state.selected_asset = 0
                                        st.session_state.analysis_mode = "deep_dive"
                                        st.rerun()
                                else:
                                    # Not a valid ticker - try searching for it
                                    search_results = search_ticker_func(ticker_input)
                                    if search_results:
                                        # Found matches - store for selection
                                        st.session_state.ticker_suggestions = search_results
                                    else:
                                        st.error(f"Could not find '{ticker_input}'. Try entering the exact ticker symbol (e.g., AAPL, SIE.DE, BTC-USD).")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please enter a ticker symbol")
        
        with col_status:
            # Show status badge if asset is selected
            if 'scan_results' in st.session_state and st.session_state.scan_results:
                results = st.session_state.scan_results
                if results:
                    selected = results[st.session_state.selected_asset]
                    regime_emoji = selected['signal'].split()[0] if selected.get('signal') else "⚪"
                    score = int(selected.get('criticality_score', 0))
                    
                    # Color code based on score
                    if score > 80:
                        badge_color = "#FF4040"
                    elif score > 60:
                        badge_color = "#FF6600"
                    else:
                        badge_color = "#00C864"
                    
                    badge_html = f'<div style="text-align: center; padding: 8px; background: rgba(102, 126, 234, 0.1); border-radius: 8px; border: 1px solid #667eea;"><span style="font-size: 0.85rem; color: #888;">Active</span><br><span style="font-size: 1.1rem; font-weight: 600;">{selected["symbol"]}</span> <span style="font-size: 1.3rem;">{regime_emoji}</span><br><span style="color: {badge_color}; font-weight: 700;">{score}</span></div>'
                    st.markdown(badge_html, unsafe_allow_html=True)


def render_education_landing(run_analysis_func: Callable) -> None:
    """
    Render the landing page when no asset is selected.
    
    Shows:
        - How SOC works (brief)
        - Top Movers / Risk List (teaser)
    
    Args:
        run_analysis_func: Function to run analysis on ticker(s)
    """
    st.markdown("### Welcome to the SOC Market Seismograph")
    
    st.markdown("""
    <div style="background: rgba(102, 126, 234, 0.1); border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
        <h4>🔬 How It Works</h4>
        <p style="font-size: 0.95rem; line-height: 1.6;">
            Self-Organized Criticality (SOC) is a physics concept applied to financial markets. 
            This app analyzes volatility patterns and trend deviations to classify assets into 
            <b>5 regime states</b> (Dormant → Stable → Active → High Energy → Critical).
        </p>
        <p style="font-size: 0.95rem; line-height: 1.6;">
            The <b>Criticality Score (0-100)</b> indicates systemic stress levels. Higher scores 
            suggest increased volatility and potential instability.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    <div style="font-size: 0.95rem; line-height: 1.8; margin-left: 1rem;">
        1. <b>Type a ticker</b> in the search box above (e.g., AAPL, TSLA, BTC-USD)<br>
        2. Analysis runs automatically<br>
        3. Explore <b>Deep Dive</b> (signals, charts) or <b>Simulation</b> (backtesting)<br>
        4. Switch assets instantly using the search bar
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Quick add popular assets
    st.markdown("### 📊 Popular Assets")
    
    col1, col2, col3, col4 = st.columns(4)
    
    popular_tickers = [
        ("AAPL", "Apple"),
        ("NVDA", "NVIDIA"),
        ("BTC-USD", "Bitcoin"),
        ("^GSPC", "S&P 500")
    ]
    
    for i, (ticker, name) in enumerate(popular_tickers):
        with [col1, col2, col3, col4][i]:
            if st.button(f"{ticker}\n{name}", key=f"quick_{ticker}", use_container_width=True):
                with st.spinner(f"Analyzing {name}..."):
                    try:
                        results = run_analysis_func([ticker])
                        if results and len(results) > 0:
                            st.session_state.current_ticker = ticker
                            st.session_state.scan_results = results
                            st.session_state.selected_asset = 0
                            st.session_state.analysis_mode = "deep_dive"
                            st.rerun()
                        else:
                            st.error(f"Could not analyze {ticker}")
                    except Exception as e:
                        st.error(f"Error analyzing {ticker}: {str(e)}")

