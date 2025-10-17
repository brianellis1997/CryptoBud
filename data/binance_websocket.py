import websocket
import json
import threading
import time
from datetime import datetime
import pandas as pd

class BinanceWebSocketClient:
    def __init__(self, symbols=None):
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT']

        self.symbols = [s.lower() for s in symbols]
        self.prices = {}
        self.ws = None
        self.running = False
        self.callbacks = []

        for symbol in self.symbols:
            self.prices[symbol] = {
                'price': 0.0,
                'timestamp': None,
                'volume': 0.0,
                'high': 0.0,
                'low': 0.0
            }

    def _build_stream_url(self):
        streams = [f"{symbol}@ticker" for symbol in self.symbols]
        stream_string = '/'.join(streams)
        url = f"wss://stream.binance.com:9443/stream?streams={stream_string}"
        return url

    def on_message(self, ws, message):
        try:
            data = json.loads(message)

            if 'data' in data:
                ticker = data['data']
                symbol = ticker['s'].lower()

                self.prices[symbol] = {
                    'price': float(ticker['c']),
                    'timestamp': datetime.fromtimestamp(ticker['E'] / 1000),
                    'volume': float(ticker['v']),
                    'high': float(ticker['h']),
                    'low': float(ticker['l']),
                    'change_24h': float(ticker['P'])
                }

                for callback in self.callbacks:
                    callback(symbol, self.prices[symbol])

        except Exception as e:
            print(f"Error processing message: {e}")

    def on_error(self, ws, error):
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        self.running = False

    def on_open(self, ws):
        print(f"WebSocket connection opened for {len(self.symbols)} symbols")
        self.running = True

    def start(self):
        url = self._build_stream_url()

        self.ws = websocket.WebSocketApp(
            url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )

        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def stop(self):
        if self.ws:
            self.ws.close()
        self.running = False

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def get_current_prices(self):
        data = []

        coin_map = {
            'btcusdt': ('bitcoin', 'BTC'),
            'ethusdt': ('ethereum', 'ETH'),
            'xrpusdt': ('ripple', 'XRP'),
            'adausdt': ('cardano', 'ADA'),
            'solusdt': ('solana', 'SOL')
        }

        for symbol, price_data in self.prices.items():
            if price_data['price'] > 0 and symbol in coin_map:
                coin_id, coin_symbol = coin_map[symbol]
                data.append({
                    'coin_id': coin_id,
                    'symbol': coin_symbol,
                    'price': price_data['price'],
                    'volume_24h': price_data['volume'],
                    'change_24h': price_data.get('change_24h', 0),
                    'high_24h': price_data['high'],
                    'low_24h': price_data['low'],
                    'timestamp': price_data['timestamp']
                })

        return pd.DataFrame(data) if data else pd.DataFrame()

    def wait_for_data(self, timeout=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if any(p['price'] > 0 for p in self.prices.values()):
                return True
            time.sleep(0.1)
        return False


class RealTimePriceTracker:
    def __init__(self, symbols=None):
        self.client = BinanceWebSocketClient(symbols)
        self.price_history = {symbol: [] for symbol in self.client.symbols}
        self.max_history = 1000

        self.client.add_callback(self._record_price)

    def _record_price(self, symbol, price_data):
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append({
            'timestamp': price_data['timestamp'],
            'price': price_data['price']
        })

        if len(self.price_history[symbol]) > self.max_history:
            self.price_history[symbol] = self.price_history[symbol][-self.max_history:]

    def start(self):
        self.client.start()
        self.client.wait_for_data()

    def stop(self):
        self.client.stop()

    def get_price_history(self, symbol, last_n=None):
        history = self.price_history.get(symbol.lower(), [])

        if last_n:
            history = history[-last_n:]

        if history:
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df

        return pd.DataFrame()


if __name__ == "__main__":
    print("Testing Binance WebSocket Real-Time Prices...")
    print("Press Ctrl+C to stop\n")

    def price_update_callback(symbol, price_data):
        print(f"{symbol.upper()}: ${price_data['price']:,.2f} | "
              f"24h Change: {price_data.get('change_24h', 0):+.2f}% | "
              f"{price_data['timestamp'].strftime('%H:%M:%S')}")

    client = BinanceWebSocketClient()
    client.add_callback(price_update_callback)
    client.start()

    client.wait_for_data()

    try:
        while True:
            time.sleep(5)
            df = client.get_current_prices()
            if not df.empty:
                print("\n" + "="*60)
                print("Current Snapshot:")
                print(df[['symbol', 'price', 'change_24h']])
                print("="*60 + "\n")
    except KeyboardInterrupt:
        print("\nStopping...")
        client.stop()
