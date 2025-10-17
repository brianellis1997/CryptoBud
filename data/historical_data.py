import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI
import time

class HistoricalDataFetcher:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.supported_coins = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'ripple': 'XRP',
            'cardano': 'ADA',
            'solana': 'SOL',
            'polkadot': 'DOT'
        }

    def fetch_historical_data(self, coin_id, days=365):
        try:
            print(f"Fetching {days} days of historical data for {coin_id}...")

            data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=days,
                interval='daily' if days > 90 else 'hourly'
            )

            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])

            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
            market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')

            df = prices.merge(volumes, on='timestamp', how='left')
            df = df.merge(market_caps, on='timestamp', how='left')

            df['coin_id'] = coin_id
            df['symbol'] = self.supported_coins[coin_id]

            df = df.sort_values('timestamp').reset_index(drop=True)

            print(f"Successfully fetched {len(df)} data points for {coin_id}")
            return df

        except Exception as e:
            print(f"Error fetching historical data for {coin_id}: {e}")
            return pd.DataFrame()

    def fetch_all_coins(self, days=365):
        all_data = []

        for coin_id in self.supported_coins.keys():
            df = self.fetch_historical_data(coin_id, days)
            if not df.empty:
                all_data.append(df)
            time.sleep(1.5)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        return pd.DataFrame()

    def add_technical_indicators(self, df):
        df = df.copy()

        df['returns'] = df['price'].pct_change()

        df['sma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
        df['sma_30'] = df['price'].rolling(window=30, min_periods=1).mean()

        df['ema_12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['price'].ewm(span=26, adjust=False).mean()

        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['volatility_7d'] = df['returns'].rolling(window=7, min_periods=1).std()

        df = df.fillna(method='bfill').fillna(0)

        return df

    def save_data(self, df, filename):
        try:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

    def load_data(self, filename):
        try:
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"Loaded {len(df)} records from {filename}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    fetcher = HistoricalDataFetcher()

    print("Fetching 180 days of historical data for all coins...")
    df = fetcher.fetch_all_coins(days=180)

    if not df.empty:
        print(f"\nTotal records fetched: {len(df)}")
        print(f"\nData shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:\n{df.head()}")

        print("\nAdding technical indicators...")
        df_with_indicators = fetcher.add_technical_indicators(df)

        filename = 'data/historical_crypto_data.csv'
        fetcher.save_data(df_with_indicators, filename)

        print(f"\nSample of data with indicators:\n{df_with_indicators.head()}")
    else:
        print("No data fetched!")
