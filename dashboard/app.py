import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

from data.fetch_crypto_data import CryptoDataFetcher
from models.predict import CryptoPricePredictor

st.set_page_config(
    page_title="CryptoBud - Real-Time Crypto Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_predictors(coins):
    predictors = {}
    for coin in coins:
        try:
            predictor = CryptoPricePredictor(coin)
            predictor.load_model_and_scaler()
            predictors[coin] = predictor
        except Exception as e:
            st.warning(f"Could not load model for {coin}: {e}")
    return predictors

@st.cache_data(ttl=60)
def fetch_current_prices(coins):
    fetcher = CryptoDataFetcher()
    return fetcher.get_current_prices(coins)

@st.cache_data(ttl=300)
def fetch_historical_data(coin_id, days=7):
    fetcher = CryptoDataFetcher()
    return fetcher.get_minute_prices(coin_id, days=days)

def create_price_chart(historical_df, predictions_df, coin_symbol, show_predictions=True):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{coin_symbol} Price & Predictions', 'Volume'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    fig.add_trace(
        go.Scatter(
            x=historical_df['timestamp'],
            y=historical_df['price'],
            name='Actual Price',
            line=dict(color='#2E86DE', width=2),
            mode='lines'
        ),
        row=1, col=1
    )

    if show_predictions and predictions_df is not None and not predictions_df.empty:
        last_actual_price = historical_df['price'].iloc[-1]
        last_timestamp = historical_df['timestamp'].iloc[-1]

        pred_timestamps = [last_timestamp] + predictions_df['timestamp'].tolist()
        pred_prices = [last_actual_price] + predictions_df['predicted_price'].tolist()

        fig.add_trace(
            go.Scatter(
                x=pred_timestamps,
                y=pred_prices,
                name='Predicted Price',
                line=dict(color='#FF6B6B', width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=8)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=pred_timestamps,
                y=pred_prices,
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.1)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    if 'volume' in historical_df.columns:
        fig.add_trace(
            go.Bar(
                x=historical_df['timestamp'],
                y=historical_df['volume'],
                name='Volume',
                marker_color='#54A0FF',
                opacity=0.5
            ),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_layout(
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_delta_chart(predictions_df, current_price):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=predictions_df['step'],
        y=predictions_df['price_change_pct'],
        marker_color=['#10AC84' if x > 0 else '#EE5A6F' for x in predictions_df['price_change_pct']],
        text=[f"{x:+.2f}%" for x in predictions_df['price_change_pct']],
        textposition='outside',
        name='Price Change %'
    ))

    fig.update_layout(
        title="Predicted Price Change (%)",
        xaxis_title="Prediction Step",
        yaxis_title="Change (%)",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )

    return fig

def main():
    st.title("📈 CryptoBud - Real-Time Crypto Dashboard")
    st.markdown("### Live cryptocurrency prices with AI-powered predictions")

    st.sidebar.header("Settings")

    available_coins = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH',
        'ripple': 'XRP',
        'cardano': 'ADA',
        'solana': 'SOL'
    }

    selected_coin = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=list(available_coins.keys()),
        format_func=lambda x: f"{available_coins[x]} ({x.capitalize()})"
    )

    timeframe = st.sidebar.selectbox(
        "Historical Timeframe",
        options=[1, 3, 7, 14, 30],
        format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
        index=2
    )

    show_predictions = st.sidebar.checkbox("Show Predictions", value=True)
    auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", value=False)

    refresh_button = st.sidebar.button("🔄 Refresh Data")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "CryptoBud uses LSTM neural networks to predict cryptocurrency prices "
        "based on historical data. Predictions are for educational purposes only."
    )

    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    if auto_refresh:
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
        if time_since_refresh >= 60 or refresh_button:
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    with st.spinner("Loading data..."):
        try:
            current_prices_df = fetch_current_prices([selected_coin])

            if not current_prices_df.empty:
                coin_data = current_prices_df[current_prices_df['coin_id'] == selected_coin].iloc[0]

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label=f"{available_coins[selected_coin]} Price",
                        value=f"${coin_data['price']:,.2f}",
                        delta=f"{coin_data['change_24h']:.2f}%"
                    )

                with col2:
                    st.metric(
                        label="Market Cap",
                        value=f"${coin_data['market_cap']/1e9:.2f}B"
                    )

                with col3:
                    st.metric(
                        label="24h Volume",
                        value=f"${coin_data['volume_24h']/1e9:.2f}B"
                    )

                with col4:
                    st.metric(
                        label="Last Updated",
                        value=coin_data['timestamp'].strftime("%H:%M:%S")
                    )

            historical_df = fetch_historical_data(selected_coin, days=timeframe)

            predictions_df = None
            if show_predictions:
                try:
                    with st.spinner("Generating predictions..."):
                        predictors = load_predictors([selected_coin])

                        if selected_coin in predictors:
                            predictor = predictors[selected_coin]

                            price_data = historical_df['price'].values[-90:]
                            predictions = predictor.predict_from_data(price_data)

                            current_price = float(historical_df['price'].iloc[-1])
                            last_timestamp = historical_df['timestamp'].iloc[-1]

                            predictions_df = pd.DataFrame({
                                'step': range(1, len(predictions) + 1),
                                'predicted_price': predictions,
                                'timestamp': [
                                    last_timestamp + timedelta(hours=i)
                                    for i in range(1, len(predictions) + 1)
                                ]
                            })

                            predictions_df['current_price'] = current_price
                            predictions_df['price_change'] = predictions_df['predicted_price'] - current_price
                            predictions_df['price_change_pct'] = (predictions_df['price_change'] / current_price) * 100

                except Exception as e:
                    st.warning(f"Predictions not available: {e}")
                    show_predictions = False

            st.markdown("---")

            chart = create_price_chart(
                historical_df,
                predictions_df,
                available_coins[selected_coin],
                show_predictions
            )
            st.plotly_chart(chart, use_container_width=True)

            if show_predictions and predictions_df is not None:
                st.markdown("---")
                st.subheader("📊 Prediction Analysis")

                col1, col2 = st.columns([2, 1])

                with col1:
                    delta_chart = create_delta_chart(predictions_df, current_price)
                    st.plotly_chart(delta_chart, use_container_width=True)

                with col2:
                    st.markdown("#### Next 5 Predictions")

                    for idx, row in predictions_df.iterrows():
                        color = "🟢" if row['price_change_pct'] > 0 else "🔴"
                        st.markdown(
                            f"{color} **Step {row['step']}**: ${row['predicted_price']:.2f} "
                            f"({row['price_change_pct']:+.2f}%)"
                        )

                    avg_change = predictions_df['price_change_pct'].mean()
                    trend = "BULLISH 📈" if avg_change > 0 else "BEARISH 📉"
                    st.markdown(f"### Trend: {trend}")
                    st.markdown(f"**Avg Change**: {avg_change:+.2f}%")

        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Make sure models are trained. Run: `python models/train_model.py`")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "Built with Streamlit & TensorFlow | Data from CoinGecko"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
