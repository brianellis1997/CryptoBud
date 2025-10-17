import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from lstm_model import CryptoLSTMModel
from utils.preprocessing import DataPreprocessor
from data.fetch_crypto_data import CryptoDataFetcher
from datetime import datetime, timedelta

class CryptoPricePredictor:
    def __init__(self, coin_id):
        self.coin_id = coin_id
        self.model = None
        self.preprocessor = DataPreprocessor(sequence_length=90)
        self.model_loaded = False

    def load_model_and_scaler(self, model_path=None, scaler_path=None):
        if model_path is None:
            model_path = f'saved_models/{self.coin_id}_lstm.keras'
        if scaler_path is None:
            scaler_path = f'saved_models/{self.coin_id}_scaler.pkl'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        self.model = CryptoLSTMModel(sequence_length=90, n_features=1, prediction_steps=5)
        self.model.load_model(model_path)

        self.preprocessor.load_scaler(scaler_path)

        self.model_loaded = True
        print(f"✓ Model and scaler loaded for {self.coin_id}")

    def predict_from_data(self, price_data):
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model_and_scaler() first.")

        if len(price_data) < 90:
            raise ValueError(f"Need at least 90 data points, got {len(price_data)}")

        X = self.preprocessor.prepare_for_prediction(price_data)

        predictions_scaled = self.model.predict(X)

        predictions = []
        for i in range(predictions_scaled.shape[1]):
            pred = self.preprocessor.inverse_transform_predictions(
                predictions_scaled[0, i:i+1]
            )
            predictions.append(float(pred[0, 0]))

        return np.array(predictions)

    def predict_realtime(self, days_lookback=90):
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model_and_scaler() first.")

        print(f"Fetching last {days_lookback} days of data for {self.coin_id}...")
        fetcher = CryptoDataFetcher()
        df = fetcher.get_minute_prices(self.coin_id, days=days_lookback)

        if df.empty:
            raise ValueError(f"No data fetched for {self.coin_id}")

        df = df.sort_values('timestamp').reset_index(drop=True)

        price_data = df['price'].values[-90:]

        predictions = self.predict_from_data(price_data)

        current_price = float(price_data[-1])
        prediction_info = {
            'coin_id': self.coin_id,
            'current_price': current_price,
            'predictions': predictions.tolist(),
            'prediction_timestamps': [
                (datetime.now() + timedelta(minutes=i*15)).isoformat()
                for i in range(1, len(predictions)+1)
            ],
            'timestamp': datetime.now().isoformat()
        }

        return prediction_info

    def get_prediction_dataframe(self, prediction_info):
        df = pd.DataFrame({
            'step': range(1, len(prediction_info['predictions']) + 1),
            'predicted_price': prediction_info['predictions'],
            'timestamp': prediction_info['prediction_timestamps']
        })

        df['current_price'] = prediction_info['current_price']
        df['price_change'] = df['predicted_price'] - df['current_price']
        df['price_change_pct'] = (df['price_change'] / df['current_price']) * 100

        return df


class MultiCoinPredictor:
    def __init__(self, coin_ids):
        self.coin_ids = coin_ids
        self.predictors = {}

        for coin_id in coin_ids:
            try:
                predictor = CryptoPricePredictor(coin_id)
                predictor.load_model_and_scaler()
                self.predictors[coin_id] = predictor
                print(f"✓ Loaded predictor for {coin_id}")
            except Exception as e:
                print(f"✗ Failed to load predictor for {coin_id}: {e}")

    def predict_all(self):
        predictions = {}

        for coin_id, predictor in self.predictors.items():
            try:
                pred_info = predictor.predict_realtime()
                predictions[coin_id] = pred_info
                print(f"✓ Generated predictions for {coin_id}")
            except Exception as e:
                print(f"✗ Failed to predict for {coin_id}: {e}")
                predictions[coin_id] = None

        return predictions

    def get_summary(self, predictions):
        summary = []

        for coin_id, pred_info in predictions.items():
            if pred_info is not None:
                current = pred_info['current_price']
                next_pred = pred_info['predictions'][0]
                change_pct = ((next_pred - current) / current) * 100

                summary.append({
                    'coin': coin_id,
                    'current_price': current,
                    'next_prediction': next_pred,
                    'change_pct': change_pct,
                    'direction': 'UP' if change_pct > 0 else 'DOWN'
                })

        return pd.DataFrame(summary)


if __name__ == "__main__":
    print("CryptoBud - Real-time Price Prediction")
    print("="*60)

    coin = 'bitcoin'

    try:
        predictor = CryptoPricePredictor(coin)

        predictor.load_model_and_scaler()

        print(f"\nGenerating predictions for {coin}...")
        prediction_info = predictor.predict_realtime(days_lookback=90)

        print(f"\nCurrent Price: ${prediction_info['current_price']:.2f}")
        print("\nPredicted Prices (next 5 steps):")
        for i, pred in enumerate(prediction_info['predictions'], 1):
            change = pred - prediction_info['current_price']
            change_pct = (change / prediction_info['current_price']) * 100
            print(f"  Step {i}: ${pred:.2f} ({change_pct:+.2f}%)")

        df = predictor.get_prediction_dataframe(prediction_info)
        print("\nPrediction DataFrame:")
        print(df)

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure to train the model first by running:")
        print("  python models/train_model.py")
