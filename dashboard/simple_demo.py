import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from data.fetch_crypto_data import CryptoDataFetcher

st.set_page_config(
    page_title="CryptoBud - Real-Time Demo",
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

@st.cache_data(ttl=30)
def fetch_current_prices():
    fetcher = CryptoDataFetcher()
    return fetcher.get_current_prices()

@st.cache_data(ttl=120)
def fetch_historical_prices(coin_id, days=1):
    fetcher = CryptoDataFetcher()
    return fetcher.get_minute_prices(coin_id, days=days)

def create_live_chart(df, coin_symbol):
    fig = go.Figure()

    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['price'],
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

def create_multi_coin_chart(coins_data, timeframe):
    fig = go.Figure()

    colors = {
        'bitcoin': '#F7931A',
        'ethereum': '#627EEA',
        'ripple': '#23292F',
        'cardano': '#0033AD',
        'solana': '#14F195'
    }

    for coin_id, data in coins_data.items():
        if not data.empty:
            coin_symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else coin_id.upper()

            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['price'],
                    name=coin_symbol,
                    line=dict(
                        color=colors.get(coin_id, '#888888'),
                        width=2
                    ),
                    mode='lines',
                    hovertemplate='%{y:$,.2f}<extra></extra>'
                )
            )

    fig.update_layout(
        title=f'All Cryptocurrencies - {timeframe} Day Comparison',
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )

    return fig

def create_normalized_multi_coin_chart(coins_data, timeframe):
    fig = go.Figure()

    colors = {
        'bitcoin': '#F7931A',
        'ethereum': '#627EEA',
        'ripple': '#23292F',
        'cardano': '#0033AD',
        'solana': '#14F195'
    }

    for coin_id, data in coins_data.items():
        if not data.empty:
            coin_symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else coin_id.upper()

            first_price = data['price'].iloc[0]
            normalized_prices = ((data['price'] / first_price) - 1) * 100

            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=normalized_prices,
                    name=coin_symbol,
                    line=dict(
                        color=colors.get(coin_id, '#888888'),
                        width=2
                    ),
                    mode='lines',
                    hovertemplate='%{y:+.2f}%<extra></extra>'
                )
            )

    fig.update_layout(
        title=f'All Cryptocurrencies - Normalized Performance ({timeframe}d)',
        xaxis_title="Time",
        yaxis_title="Change from Start (%)",
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )

    return fig

def main():
    st.title("📈 CryptoBud - Real-Time Demo")
    st.markdown("### Live cryptocurrency prices via CoinGecko API")

    st.sidebar.header("Settings")

    available_coins = {
        'bitcoin': 'BTC - Bitcoin',
        'ethereum': 'ETH - Ethereum',
        'ripple': 'XRP - Ripple',
        'cardano': 'ADA - Cardano',
        'solana': 'SOL - Solana'
    }

    selected_coin = st.sidebar.selectbox(
        "Select Cryptocurrency",
        options=list(available_coins.keys()),
        format_func=lambda x: available_coins[x]
    )

    timeframe = st.sidebar.selectbox(
        "Historical Timeframe",
        options=[1, 3, 7],
        format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
        index=0
    )

    chart_mode = st.sidebar.radio(
        "Chart Mode",
        options=["Single Coin", "All Coins", "Normalized Comparison"],
        help="Single Coin: Show one coin at a time\n"
             "All Coins: Show all coins on same chart\n"
             "Normalized: Compare % change from start"
    )

    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    refresh_button = st.sidebar.button("🔄 Refresh Now")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Demo Mode** - No training required!\n\n"
        "Prices update every 30 seconds.\n\n"
        "Train models to enable AI predictions:\n"
        "`python models/train_model.py`"
    )

    with st.spinner("Loading data..."):
        current_df = fetch_current_prices()

        if not current_df.empty:
            coin_data = current_df[current_df['coin_id'] == selected_coin].iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label=f"{coin_data['symbol']} Price",
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
                    label="Last Update",
                    value=coin_data['timestamp'].strftime("%H:%M:%S")
                )

            st.markdown("---")

            if chart_mode == "Single Coin":
                historical_df = fetch_historical_prices(selected_coin, days=timeframe)

                if not historical_df.empty:
                    chart = create_live_chart(historical_df, coin_data['symbol'])
                    st.plotly_chart(chart, use_container_width=True)

            elif chart_mode in ["All Coins", "Normalized Comparison"]:
                with st.spinner("Loading data for all coins..."):
                    all_coins_data = {}

                    for coin_id in available_coins.keys():
                        coin_df = fetch_historical_prices(coin_id, days=timeframe)
                        if not coin_df.empty:
                            all_coins_data[coin_id] = coin_df

                    if all_coins_data:
                        if chart_mode == "All Coins":
                            chart = create_multi_coin_chart(all_coins_data, timeframe)
                        else:
                            chart = create_normalized_multi_coin_chart(all_coins_data, timeframe)

                        st.plotly_chart(chart, use_container_width=True)

                        st.info(
                            "💡 **Tip**: Normalized view shows % change from the start, "
                            "making it easier to compare performance across different price ranges."
                        )

                st.markdown("---")
                st.subheader("📊 All Cryptocurrencies")

                cols = st.columns(5)
                for idx, (coin_id, name) in enumerate(available_coins.items()):
                    c_data = current_df[current_df['coin_id'] == coin_id]

                    if not c_data.empty:
                        c_data = c_data.iloc[0]
                        with cols[idx]:
                            change_color = "🟢" if c_data['change_24h'] >= 0 else "🔴"
                            st.markdown(
                                f"**{change_color} {c_data['symbol']}**\n\n"
                                f"${c_data['price']:,.2f}\n\n"
                                f"{c_data['change_24h']:+.2f}%"
                            )

                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    price_range = historical_df['price'].max() - historical_df['price'].min()
                    st.metric(
                        f"{timeframe}d Price Range",
                        f"${price_range:.2f}"
                    )

                with col2:
                    avg_price = historical_df['price'].mean()
                    st.metric(
                        f"{timeframe}d Avg Price",
                        f"${avg_price:,.2f}"
                    )

                with col3:
                    volatility = historical_df['price'].std()
                    st.metric(
                        f"{timeframe}d Volatility",
                        f"${volatility:.2f}"
                    )

    if auto_refresh:
        time.sleep(30)
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "Demo Mode - Data from CoinGecko API | Updates every 30 seconds<br>"
        "Train models for AI predictions: python models/train_model.py"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
