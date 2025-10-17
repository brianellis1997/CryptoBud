import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np

from data.binance_websocket import BinanceWebSocketClient

st.set_page_config(
    page_title="CryptoBud Demo - Real-Time Prices",
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
    .price-update {
        font-size: 2rem;
        font-weight: bold;
        animation: pulse 1s ease-in-out;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_websocket_client():
    client = BinanceWebSocketClient()
    client.start()
    client.wait_for_data(timeout=10)
    return client

def create_live_chart(price_history_df, coin_symbol):
    fig = go.Figure()

    if not price_history_df.empty:
        fig.add_trace(
            go.Scatter(
                x=price_history_df['timestamp'],
                y=price_history_df['price'],
                name='Price',
                line=dict(color='#2E86DE', width=2),
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(46, 134, 222, 0.1)'
            )
        )

    fig.update_layout(
        title=f'{coin_symbol} - Live Price Movement',
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=500,
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def main():
    st.title("📈 CryptoBud Demo - Real-Time Crypto Prices")
    st.markdown("### Live cryptocurrency prices updated EVERY SECOND via Binance WebSocket")

    st.sidebar.header("Settings")

    available_coins = {
        'btcusdt': 'BTC - Bitcoin',
        'ethusdt': 'ETH - Ethereum',
        'xrpusdt': 'XRP - Ripple',
        'adausdt': 'ADA - Cardano',
        'solusdt': 'SOL - Solana'
    }

    selected_symbol = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=list(available_coins.keys()),
        format_func=lambda x: available_coins[x]
    )

    update_interval = st.sidebar.slider(
        "Update Interval (seconds)",
        min_value=1,
        max_value=10,
        value=1,
        help="How often to refresh the dashboard"
    )

    show_mini_charts = st.sidebar.checkbox("Show All Coins", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Demo Mode")
    st.sidebar.info(
        "This is DEMO mode with real-time prices. "
        "Train models to enable AI predictions!\n\n"
        "Run: `python models/train_model.py`"
    )

    if 'websocket_client' not in st.session_state:
        with st.spinner("Connecting to Binance WebSocket..."):
            st.session_state.websocket_client = get_websocket_client()

    ws_client = st.session_state.websocket_client

    if 'price_tracker' not in st.session_state:
        st.session_state.price_tracker = {
            symbol: [] for symbol in available_coins.keys()
        }

    placeholder = st.empty()

    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

    while True:
        with placeholder.container():
            current_df = ws_client.get_current_prices()

            if not current_df.empty:
                selected_coin = current_df[
                    current_df['symbol'] == selected_symbol.replace('usdt', '').upper()
                ]

                if not selected_coin.empty:
                    coin_data = selected_coin.iloc[0]

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            label=f"{coin_data['symbol']} Price",
                            value=f"${coin_data['price']:,.2f}",
                            delta=f"{coin_data['change_24h']:.2f}%"
                        )

                    with col2:
                        st.metric(
                            label="24h High",
                            value=f"${coin_data['high_24h']:,.2f}"
                        )

                    with col3:
                        st.metric(
                            label="24h Low",
                            value=f"${coin_data['low_24h']:,.2f}"
                        )

                    with col4:
                        st.metric(
                            label="Last Update",
                            value=coin_data['timestamp'].strftime("%H:%M:%S")
                        )

                    current_price = coin_data['price']
                    if selected_symbol not in st.session_state.price_tracker:
                        st.session_state.price_tracker[selected_symbol] = []

                    st.session_state.price_tracker[selected_symbol].append({
                        'timestamp': datetime.now(),
                        'price': current_price
                    })

                    if len(st.session_state.price_tracker[selected_symbol]) > 300:
                        st.session_state.price_tracker[selected_symbol] = \
                            st.session_state.price_tracker[selected_symbol][-300:]

                    price_history = pd.DataFrame(
                        st.session_state.price_tracker[selected_symbol]
                    )

                    st.markdown("---")

                    chart = create_live_chart(
                        price_history,
                        coin_data['symbol']
                    )
                    st.plotly_chart(chart, use_container_width=True)

                    if show_mini_charts:
                        st.markdown("---")
                        st.subheader("📊 All Cryptocurrencies")

                        cols = st.columns(3)
                        for idx, (symbol, name) in enumerate(available_coins.items()):
                            coin_df = current_df[
                                current_df['symbol'] == symbol.replace('usdt', '').upper()
                            ]

                            if not coin_df.empty:
                                c_data = coin_df.iloc[0]
                                with cols[idx % 3]:
                                    change_color = "🟢" if c_data['change_24h'] >= 0 else "🔴"
                                    st.markdown(
                                        f"**{change_color} {c_data['symbol']}**\n\n"
                                        f"${c_data['price']:,.2f}\n\n"
                                        f"{c_data['change_24h']:+.2f}%"
                                    )

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if price_history.shape[0] > 1:
                            price_change = price_history['price'].iloc[-1] - price_history['price'].iloc[0]
                            price_change_pct = (price_change / price_history['price'].iloc[0]) * 100
                            st.metric(
                                "Session Change",
                                f"${price_change:+,.2f}",
                                f"{price_change_pct:+.2f}%"
                            )

                    with col2:
                        if price_history.shape[0] > 0:
                            volatility = price_history['price'].std()
                            st.metric(
                                "Session Volatility",
                                f"${volatility:.2f}"
                            )

                    with col3:
                        st.metric(
                            "Data Points",
                            len(st.session_state.price_tracker[selected_symbol])
                        )

        if not auto_refresh:
            break

        time.sleep(update_interval)
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "Demo Mode - Real-time data from Binance WebSocket | "
        "Train models to enable AI predictions"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
