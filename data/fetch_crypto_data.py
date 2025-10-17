import requests
import pandas as pd
import time
from datetime import datetime
from pycoingecko import CoinGeckoAPI

class CryptoDataFetcher:
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

    def get_current_prices(self, coin_ids=None):
        if coin_ids is None:
            coin_ids = list(self.supported_coins.keys())

        try:
            prices = self.cg.get_price(
                ids=coin_ids,
                vs_currencies='usd',
                include_24hr_change=True,
                include_market_cap=True,
                include_24hr_vol=True
            )

            data = []
            for coin_id in coin_ids:
                if coin_id in prices:
                    data.append({
                        'coin_id': coin_id,
                        'symbol': self.supported_coins[coin_id],
                        'price': prices[coin_id]['usd'],
                        'market_cap': prices[coin_id].get('usd_market_cap', 0),
                        'volume_24h': prices[coin_id].get('usd_24h_vol', 0),
                        'change_24h': prices[coin_id].get('usd_24h_change', 0),
                        'timestamp': datetime.now()
                    })

            return pd.DataFrame(data)

        except Exception as e:
            print(f"Error fetching prices: {e}")
            return pd.DataFrame()

    def get_realtime_feed(self, coin_ids=None, interval_seconds=60):
        if coin_ids is None:
            coin_ids = list(self.supported_coins.keys())

        all_data = []

        try:
            while True:
                df = self.get_current_prices(coin_ids)
                if not df.empty:
                    all_data.append(df)
                    yield df

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("Stopped real-time feed")
            if all_data:
                return pd.concat(all_data, ignore_index=True)

    def get_minute_prices(self, coin_id, days=1):
        try:
            data = self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=days
            )

            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
            prices_df['coin_id'] = coin_id
            prices_df['symbol'] = self.supported_coins[coin_id]

            return prices_df

        except Exception as e:
            print(f"Error fetching minute prices for {coin_id}: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df, filename):
        try:
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")


if __name__ == "__main__":
    fetcher = CryptoDataFetcher()

    print("Fetching current prices for all supported coins...")
    df = fetcher.get_current_prices()
    print(df)

    print("\nFetching minute-level data for Bitcoin (last 24 hours)...")
    btc_data = fetcher.get_minute_prices('bitcoin', days=1)
    print(f"Retrieved {len(btc_data)} data points")
    print(btc_data.head())
