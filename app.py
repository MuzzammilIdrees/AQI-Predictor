"""
AQI Predictor Dashboard - Modern Interactive UI
A beautiful, user-friendly dashboard for air quality predictions
"""
import datetime as dt
import os
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import settings
from src.data_fetch import fetch_air_quality
from src.feature_engineering import build_features
from src.feature_store import FeatureStore
from src.predict import load_model, load_shap, predict, tag_hazard, top_shap_contributors, get_lime_explanation
from src.train import train_models


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="AQI Predictor | Air Quality Forecast",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS Styling - Dark Theme with Gradient Accents
# ============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #a0a0b0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    /* AQI Level Colors */
    .aqi-good { color: #00e676 !important; }
    .aqi-moderate { color: #ffeb3b !important; }
    .aqi-unhealthy { color: #ff9800 !important; }
    .aqi-very-unhealthy { color: #f44336 !important; }
    .aqi-hazardous { color: #9c27b0 !important; }
    
    /* Badge styling */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-good { background: rgba(0,230,118,0.2); color: #00e676; }
    .badge-moderate { background: rgba(255,235,59,0.2); color: #ffeb3b; }
    .badge-unhealthy { background: rgba(255,152,0,0.2); color: #ff9800; }
    .badge-very-unhealthy { background: rgba(244,67,54,0.2); color: #f44336; }
    
    /* Card container */
    .info-card {
        background: rgba(30, 30, 46, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 30, 46, 0.6);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #e0e0e0 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
    }
    
    /* Health recommendation cards */
    .health-card {
        background: linear-gradient(145deg, #1e3a5f 0%, #1a2f4a 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #667eea;
        margin-bottom: 0.8rem;
    }
    
    /* Animation for metrics */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.5s ease-out forwards;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Utility Functions
# ============================================================================
if hasattr(st, "cache_data"):
    cache_fn = st.cache_data
else:
    cache_fn = st.cache


@cache_fn(show_spinner=False, ttl=300)
def load_feature_store():
    store = FeatureStore(settings.feature_store_path)
    return store.load()


def ensure_trained_model(force_retrain=False):
    """Ensure model and SHAP explainer are available and compatible.
    
    If SHAP fails to load (version mismatch), automatically retrain to regenerate.
    """
    model_path = settings.model_path
    shap_path = settings.shap_path
    
    # Check if model exists
    model_exists = model_path.exists()
    
    # Check if SHAP is loadable
    shap_works = False
    if shap_path.exists() and not force_retrain:
        try:
            shap_payload = load_shap()
            if shap_payload is not None and "explainer" in shap_payload:
                shap_works = True
        except Exception:
            shap_works = False
    
    # If both exist and work, we're good
    if model_exists and shap_works and not force_retrain:
        return True
    
    # Otherwise, need to train/retrain
    if not model_exists:
        st.warning("⚡ No trained model found. Training a new model with backfilled data...")
    elif not shap_works:
        st.info("🔄 Regenerating SHAP explainer for this environment...")
    
    store = FeatureStore(settings.feature_store_path)
    df = store.load()
    if df is None or df.empty:
        raw = fetch_air_quality(settings.default_city, past_days=10, forecast_days=1)
        features = build_features(raw)
        store.append(features)
        df = features
    
    with st.spinner("Training model and generating SHAP explainer... This may take a minute."):
        train_models(df)
    
    st.success("✅ Model and SHAP explainer ready!")
    return True


@cache_fn(show_spinner=False, ttl=60)
def fetch_and_predict(city: str):
    raw = fetch_air_quality(city, past_days=1, forecast_days=4)
    features = build_features(raw)
    preds = predict(features)
    return raw, preds


def get_aqi_color(aqi: float) -> str:
    """Return color based on AQI level."""
    if aqi < 50:
        return "#00e676"  # Good - Green
    elif aqi < 100:
        return "#ffeb3b"  # Moderate - Yellow
    elif aqi < 150:
        return "#ff9800"  # Unhealthy for Sensitive - Orange
    elif aqi < 200:
        return "#f44336"  # Unhealthy - Red
    elif aqi < 300:
        return "#9c27b0"  # Very Unhealthy - Purple
    else:
        return "#7b1fa2"  # Hazardous - Dark Purple


def get_aqi_level(aqi: float) -> tuple:
    """Return AQI level name as (level, badge_class, emoji)."""
    if aqi < 50:
        return ("Good", "badge-good", "✅")
    elif aqi < 100:
        return ("Moderate", "badge-moderate", "🟡")
    elif aqi < 150:
        return ("Unhealthy for Sensitive Groups", "badge-unhealthy", "🟠")
    elif aqi < 200:
        return ("Unhealthy", "badge-unhealthy", "🔴")
    elif aqi < 300:
        return ("Very Unhealthy", "badge-very-unhealthy", "🟣")
    else:
        return ("Hazardous", "badge-very-unhealthy", "⛔")


def get_health_recommendations(aqi: float) -> list:
    """Return health recommendations based on AQI level."""
    if aqi < 50:
        return [
            ("🏃 Outdoor Activities", "Air quality is excellent! Perfect for outdoor exercise and activities."),
            ("🪟 Ventilation", "Feel free to open windows and enjoy fresh air."),
            ("👨‍👩‍👧‍👦 All Groups", "No health concerns for any population group.")
        ]
    elif aqi < 100:
        return [
            ("🏃 Outdoor Activities", "Generally acceptable for most people."),
            ("⚠️ Sensitive Groups", "Unusually sensitive individuals may experience minor symptoms."),
            ("💡 Tip", "Consider reducing prolonged outdoor exertion if you experience symptoms.")
        ]
    elif aqi < 150:
        return [
            ("👴 Sensitive Groups", "Children, elderly, and those with respiratory issues should limit outdoor exposure."),
            ("🏠 Indoor Activities", "Consider moving activities indoors."),
            ("😷 Protection", "N95 masks recommended for sensitive individuals outdoors.")
        ]
    elif aqi < 200:
        return [
            ("🚫 Limit Outdoor Exposure", "Everyone should reduce prolonged outdoor exertion."),
            ("😷 Wear Protection", "N95 masks strongly recommended when outdoors."),
            ("🏥 Health Alert", "People with heart/lung disease, children, and elderly are at greater risk.")
        ]
    else:
        return [
            ("⛔ Health Emergency", "Serious health effects for everyone. Avoid all outdoor activities."),
            ("🏠 Stay Indoors", "Keep windows closed and use air purifiers if available."),
            ("😷 Essential Travel Only", "If you must go outside, wear N95 mask and limit exposure time.")
        ]


# ============================================================================
# Visualization Functions
# ============================================================================
def create_aqi_gauge(current_aqi: float) -> go.Figure:
    """Create an animated AQI gauge meter."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_aqi,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current AQI", 'font': {'size': 20, 'color': '#e0e0e0'}},
        number={'font': {'size': 48, 'color': get_aqi_color(current_aqi)}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 2, 'tickcolor': "#666"},
            'bar': {'color': get_aqi_color(current_aqi), 'thickness': 0.3},
            'bgcolor': "rgba(30,30,46,0.8)",
            'borderwidth': 2,
            'bordercolor': "#444",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0,230,118,0.3)'},
                {'range': [50, 100], 'color': 'rgba(255,235,59,0.3)'},
                {'range': [100, 150], 'color': 'rgba(255,152,0,0.3)'},
                {'range': [150, 200], 'color': 'rgba(244,67,54,0.3)'},
                {'range': [200, 300], 'color': 'rgba(156,39,176,0.3)'},
                {'range': [300, 500], 'color': 'rgba(123,31,162,0.3)'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': current_aqi
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e0e0e0'},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def create_forecast_chart(preds: pd.DataFrame) -> go.Figure:
    """Create an interactive forecast chart with confidence bands."""
    fig = go.Figure()
    
    # Add confidence band (simulated ±15%)
    upper = preds['prediction'] * 1.15
    lower = preds['prediction'] * 0.85
    
    fig.add_trace(go.Scatter(
        x=preds['time'],
        y=upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=preds['time'],
        y=lower,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.2)',
        name='Confidence Band',
        hoverinfo='skip'
    ))
    
    # Add main prediction line
    fig.add_trace(go.Scatter(
        x=preds['time'],
        y=preds['prediction'],
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='#667eea', width=3),
        marker=dict(size=6, color='#764ba2'),
        hovertemplate='<b>%{x}</b><br>AQI: %{y:.1f}<extra></extra>'
    ))
    
    # Add hazard threshold lines
    thresholds = [
        (100, 'Moderate', '#ffeb3b'),
        (150, 'Unhealthy', '#ff9800'),
        (200, 'Very Unhealthy', '#f44336')
    ]
    
    for threshold, label, color in thresholds:
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color=color,
            annotation_text=label,
            annotation_position="right",
            opacity=0.6
        )
    
    fig.update_layout(
        title=dict(
            text='72-Hour AQI Forecast',
            font=dict(size=20, color='#e0e0e0')
        ),
        xaxis_title='Time',
        yaxis_title='Air Quality Index (AQI)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,46,0.6)',
        font=dict(color='#e0e0e0'),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=60, r=20, t=80, b=60),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=False
        ),
        height=450
    )
    
    return fig


def create_pollutant_chart(raw: pd.DataFrame) -> go.Figure:
    """Create a multi-pollutant comparison chart."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PM2.5 Levels', 'PM10 Levels', 'Temperature', 'Humidity'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # PM2.5
    if 'pm2_5' in raw.columns:
        fig.add_trace(
            go.Scatter(x=raw['time'], y=raw['pm2_5'], name='PM2.5',
                      line=dict(color='#e74c3c', width=2)),
            row=1, col=1
        )
    
    # PM10
    if 'pm10' in raw.columns:
        fig.add_trace(
            go.Scatter(x=raw['time'], y=raw['pm10'], name='PM10',
                      line=dict(color='#3498db', width=2)),
            row=1, col=2
        )
    
    # Temperature
    if 'temperature_2m' in raw.columns:
        fig.add_trace(
            go.Scatter(x=raw['time'], y=raw['temperature_2m'], name='Temp (°C)',
                      line=dict(color='#f39c12', width=2)),
            row=2, col=1
        )
    
    # Humidity
    if 'relative_humidity_2m' in raw.columns:
        fig.add_trace(
            go.Scatter(x=raw['time'], y=raw['relative_humidity_2m'], name='Humidity (%)',
                      line=dict(color='#2ecc71', width=2)),
            row=2, col=2
        )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,46,0.6)',
        font=dict(color='#e0e0e0'),
        showlegend=False,
        height=500,
        margin=dict(l=60, r=20, t=60, b=40)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
    
    return fig


def create_shap_chart(top_feats: list) -> go.Figure:
    """Create a horizontal bar chart for SHAP feature importance."""
    if not top_feats:
        return None
    
    features = [f[0] for f in top_feats][::-1]
    importances = [f[1] for f in top_feats][::-1]
    
    colors = ['#667eea' if i % 2 == 0 else '#764ba2' for i in range(len(features))]
    
    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Feature Importance (SHAP Values)',
            font=dict(size=18, color='#e0e0e0')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,46,0.6)',
        font=dict(color='#e0e0e0'),
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='',
        margin=dict(l=150, r=20, t=60, b=40),
        height=400,
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig


# ============================================================================
# Main Application
# ============================================================================
def main():
    inject_custom_css()
    
    # ========== Header ==========
    st.markdown("""
    <div class="main-header">
        <h1>🌬️ AQI Predictor</h1>
        <p>Advanced Air Quality Forecasting with Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== Sidebar ==========
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        city = st.selectbox(
            "🌍 Select City",
            options=list(settings.city_coords.keys()),
            index=0,
            help="Choose a city for air quality prediction"
        )
        
        coords = settings.city_coords[city]
        st.caption(f"📍 Coordinates: {coords[0]:.4f}, {coords[1]:.4f}")
        
        st.markdown("---")
        
        st.markdown("### 📊 Display Options")
        show_raw_data = st.checkbox("Show Raw Data Table", value=False)
        show_pollutants = st.checkbox("Show Pollutant Details", value=True)
        auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("""
        This dashboard uses machine learning to predict 
        Air Quality Index (AQI) for the next 72 hours.
        
        Data source: Open-Meteo API
        Models: RandomForest, GradientBoosting
        """)
    
    # ========== Load Data ==========
    with st.spinner("🔄 Loading model and fetching data..."):
        ensure_trained_model()
        model = load_model()
        shap_payload = None
        if settings.shap_path.exists():
            shap_payload = load_shap()
    
    try:
        raw, preds = fetch_and_predict(city)
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        st.info("Please check your internet connection and try again.")
        return
    
    # ========== Current Status Row ==========
    current_aqi = preds.iloc[-1]['prediction'] if not preds.empty else 0
    level_info = get_aqi_level(current_aqi)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.plotly_chart(create_aqi_gauge(current_aqi), use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Air Quality Status</div>
            <div class="metric-value">{level_info[2]} {level_info[0]}</div>
            <br>
            <div class="metric-label">Last Updated</div>
            <div style="color: #e0e0e0; font-size: 1.1rem;">
                {preds.iloc[-1]['time'].strftime('%Y-%m-%d %H:%M') if not preds.empty else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Quick stats
        avg_aqi = preds['prediction'].mean()
        max_aqi = preds['prediction'].max()
        min_aqi = preds['prediction'].min()
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">72-Hour Statistics</div>
            <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                <div style="text-align: center;">
                    <div style="color: #00e676; font-size: 1.4rem; font-weight: 600;">{min_aqi:.0f}</div>
                    <div style="color: #888; font-size: 0.8rem;">MIN</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #667eea; font-size: 1.4rem; font-weight: 600;">{avg_aqi:.0f}</div>
                    <div style="color: #888; font-size: 0.8rem;">AVG</div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #f44336; font-size: 1.4rem; font-weight: 600;">{max_aqi:.0f}</div>
                    <div style="color: #888; font-size: 0.8rem;">MAX</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== Tabs for Different Views ==========
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🏥 Health Advisory", "🔬 SHAP Analysis", "📋 Data"])
    
    # ----- TAB 1: Forecast -----
    with tab1:
        st.plotly_chart(create_forecast_chart(preds), use_container_width=True)
        
        # Alerts section
        st.markdown("### ⚠️ Hazard Alerts")
        alert_rows = preds[preds['prediction'] >= 100]
        
        if alert_rows.empty:
            st.success("✅ No hazardous air quality levels predicted in the next 72 hours!")
        else:
            for _, row in alert_rows.sort_values('time').head(5).iterrows():
                hazard = tag_hazard(row['prediction'])
                color = get_aqi_color(row['prediction'])
                st.markdown(f"""
                <div class="info-card" style="border-left: 4px solid {color};">
                    <strong>{row['time'].strftime('%b %d, %H:%M')}</strong> — 
                    AQI: <span style="color: {color}; font-weight: 600;">{row['prediction']:.0f}</span> 
                    ({hazard})
                </div>
                """, unsafe_allow_html=True)
        
        # Pollutant breakdown
        if show_pollutants:
            st.markdown("### 🧪 Pollutant Levels")
            st.plotly_chart(create_pollutant_chart(raw.tail(48)), use_container_width=True)
    
    # ----- TAB 2: Health Advisory -----
    with tab2:
        st.markdown("### 🏥 Health Recommendations")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {get_aqi_color(current_aqi)}22, {get_aqi_color(current_aqi)}11); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                    border: 1px solid {get_aqi_color(current_aqi)}44;">
            <h3 style="margin: 0; color: {get_aqi_color(current_aqi)};">
                {level_info[2]} Current Level: {level_info[0]}
            </h3>
            <p style="color: #ccc; margin-top: 0.5rem;">
                Based on current AQI of {current_aqi:.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        recommendations = get_health_recommendations(current_aqi)
        
        for title, desc in recommendations:
            st.markdown(f"""
            <div class="health-card">
                <strong style="color: #667eea; font-size: 1.1rem;">{title}</strong>
                <p style="color: #ccc; margin: 0.5rem 0 0 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sensitive groups info
        with st.expander("👥 Information for Sensitive Groups"):
            st.markdown("""
            **Sensitive groups include:**
            - Children under 12 years
            - Adults over 65 years
            - People with respiratory conditions (asthma, COPD)
            - People with heart conditions
            - Pregnant women
            - Outdoor workers
            
            **Protective measures:**
            - Monitor local air quality alerts
            - Keep medication accessible
            - Use N95 or KN95 masks when outdoors
            - Consider using air purifiers indoors
            """)
    
    # ----- TAB 3: SHAP Analysis -----
    with tab3:
        st.markdown("### 🔬 Model Explainability (SHAP)")
        
        if shap_payload is not None:
            try:
                # Load feature store data for SHAP analysis (has all required columns)
                store = FeatureStore(settings.feature_store_path)
                feature_data = store.load()
                
                if feature_data is not None and not feature_data.empty:
                    # Use feature store data which has all columns needed by SHAP
                    top_feats = top_shap_contributors(shap_payload, feature_data.tail(100), max_features=10)
                    
                    if top_feats:
                        shap_chart = create_shap_chart(top_feats)
                        if shap_chart:
                            st.plotly_chart(shap_chart, use_container_width=True)
                        
                        with st.expander("📖 Understanding SHAP Values"):
                            st.markdown("""
                            **SHAP (SHapley Additive exPlanations)** values show how each feature 
                            contributes to the model's prediction.
                            
                            - **Higher bars** = More important features
                            - Features like `pm2_5` and `pm10` are typically most important for AQI
                            - Rolling averages help capture trends in air quality
                            - Time features (hour, day) capture daily patterns
                            """)
                    else:
                        st.info("📊 SHAP values computed but no significant feature importance found.")
                else:
                    st.info("📊 Feature store is empty. SHAP analysis will be available after data collection.")
                    
            except Exception as e:
                st.warning(f"⚠️ SHAP analysis unavailable: {e}")
                st.info("The model will automatically regenerate SHAP values on the next training run.")
        else:
            st.info("""
            🔄 **SHAP explainer not available**
            
            This can happen when:
            - The model was trained on a different Python version
            - SHAP package version mismatch
            
            The explainer will be regenerated on the next model training.
            """)
            
            if st.button("🔧 Retrain Model to Generate SHAP"):
                with st.spinner("Training model..."):
                    store = FeatureStore(settings.feature_store_path)
                    df = store.load()
                    if df is not None and not df.empty:
                        train_models(df)
                        st.success("✅ Model retrained! Please refresh the page.")
                        st.rerun()
        
        # LIME Explanation Section
        st.markdown("---")
        st.markdown("### 🍋 LIME Explanation (Instance-level)")
        st.caption("LIME provides explanations for individual predictions")
        
        try:
            # Use feature store data for LIME (has all required columns)
            store = FeatureStore(settings.feature_store_path)
            lime_data = store.load()
            
            if lime_data is not None and not lime_data.empty and len(lime_data) >= 10:
                # Get LIME explanation for the most recent instance
                lime_results = get_lime_explanation(model, lime_data.tail(50), instance_idx=-1, num_features=8)
                
                if lime_results:
                    lime_chart = create_shap_chart(lime_results)  # Reuse chart function
                    if lime_chart:
                        lime_chart.update_layout(title="LIME Feature Contributions (Latest Prediction)")
                        st.plotly_chart(lime_chart, use_container_width=True)
                    
                    with st.expander("📖 Understanding LIME"):
                        st.markdown("""
                        **LIME (Local Interpretable Model-agnostic Explanations)** 
                        explains individual predictions by:
                        
                        - Creating perturbations around the instance
                        - Training a simple interpretable model locally
                        - Showing which features influenced this specific prediction
                        
                        Unlike SHAP (global), LIME is **instance-specific**.
                        """)
                else:
                    st.info("📊 LIME could not generate explanation for this instance.")
            else:
                st.info("📊 Not enough data in feature store for LIME analysis (need at least 10 records).")
        except Exception as e:
            st.info(f"LIME analysis not available: {str(e)[:100]}")
    
    # ----- TAB 4: Data -----
    with tab4:
        st.markdown("### 📋 Prediction Data")
        
        # Download button
        csv = preds.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"aqi_predictions_{city}_{dt.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("#### Forecasted Values")
        display_df = preds[['time', 'prediction', 'hazard_level']].copy()
        display_df.columns = ['Time', 'Predicted AQI', 'Hazard Level']
        display_df['Predicted AQI'] = display_df['Predicted AQI'].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        if show_raw_data:
            st.markdown("#### Raw Sensor Data")
            st.dataframe(raw.tail(48), use_container_width=True, hide_index=True)
    
    # ========== Footer ==========
    st.markdown("""
    <div class="footer">
        <p>🌬️ AQI Predictor | Powered by Open-Meteo API & Scikit-Learn</p>
        <p style="font-size: 0.75rem;">Data refreshes every 5 minutes • Models retrain automatically with new data</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
