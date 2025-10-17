import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from lstm_model import CryptoLSTMModel
from utils.preprocessing import DataPreprocessor, TimeSeriesValidator
from data.historical_data import HistoricalDataFetcher
import matplotlib.pyplot as plt
import json
from datetime import datetime

class ModelTrainer:
    def __init__(self, coin_id, sequence_length=90, prediction_steps=5):
        self.coin_id = coin_id
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.preprocessor = DataPreprocessor(sequence_length=sequence_length)
        self.model = None
        self.metrics = {}

    def load_and_prepare_data(self, data_path=None):
        print(f"\n{'='*60}")
        print(f"Preparing data for {self.coin_id}")
        print(f"{'='*60}")

        if data_path and os.path.exists(data_path):
            fetcher = HistoricalDataFetcher()
            df = fetcher.load_data(data_path)
            df = df[df['coin_id'] == self.coin_id].copy()
        else:
            print("Fetching fresh historical data...")
            fetcher = HistoricalDataFetcher()
            df = fetcher.fetch_historical_data(self.coin_id, days=365)
            df = fetcher.add_technical_indicators(df)

        if df.empty:
            raise ValueError(f"No data available for {self.coin_id}")

        print(f"Total records: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        scaled_data, processed_df = self.preprocessor.prepare_data(
            df,
            target_column='price',
            feature_columns=['price']
        )

        X, y = self.preprocessor.create_sequences(
            scaled_data,
            prediction_steps=self.prediction_steps
        )

        print(f"Sequences created - X: {X.shape}, y: {y.shape}")

        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split(
            X, y, train_size=0.8
        )

        val_split = int(len(X_train) * 0.8)
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test, processed_df

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        print(f"\n{'='*60}")
        print(f"Training LSTM Model for {self.coin_id}")
        print(f"{'='*60}")

        self.model = CryptoLSTMModel(
            sequence_length=self.sequence_length,
            n_features=X_train.shape[2],
            prediction_steps=self.prediction_steps
        )

        self.model.build_model(
            lstm_units=[100, 75, 50],
            dropout_rate=0.2,
            learning_rate=0.001
        )

        print("\nModel Architecture:")
        self.model.summary()

        model_path = f'saved_models/{self.coin_id}_lstm.keras'
        os.makedirs('saved_models', exist_ok=True)

        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path
        )

        self.plot_training_history(history)

        return history

    def evaluate(self, X_test, y_test):
        print(f"\n{'='*60}")
        print(f"Evaluating Model for {self.coin_id}")
        print(f"{'='*60}")

        predictions = self.model.predict(X_test)

        y_test_first_step = y_test[:, 0]
        predictions_first_step = predictions[:, 0]

        y_test_inverse = self.preprocessor.inverse_transform_predictions(y_test_first_step)
        predictions_inverse = self.preprocessor.inverse_transform_predictions(predictions_first_step)

        metrics = TimeSeriesValidator.calculate_metrics(y_test_inverse, predictions_inverse)
        directional_acc = TimeSeriesValidator.directional_accuracy(y_test_inverse, predictions_inverse)

        metrics['Directional_Accuracy'] = directional_acc

        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        self.metrics = metrics

        return metrics, predictions, y_test

    def plot_training_history(self, history):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'{self.coin_id.upper()} - Model Loss')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.history['mae'], label='Training MAE')
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_title(f'{self.coin_id.upper()} - Mean Absolute Error')
        axes[1].set_ylabel('MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'saved_models/{self.coin_id}_training_history.png')
        print(f"\nTraining history plot saved to saved_models/{self.coin_id}_training_history.png")
        plt.close()

    def save_artifacts(self):
        scaler_path = f'saved_models/{self.coin_id}_scaler.pkl'
        self.preprocessor.save_scaler(scaler_path)

        metrics_path = f'saved_models/{self.coin_id}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")


def train_all_coins(coins=None, epochs=100):
    if coins is None:
        coins = ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana']

    results = {}

    for coin in coins:
        try:
            print(f"\n\n{'#'*60}")
            print(f"# TRAINING MODEL FOR {coin.upper()}")
            print(f"{'#'*60}")

            trainer = ModelTrainer(coin, sequence_length=90, prediction_steps=5)

            X_train, X_val, X_test, y_train, y_val, y_test, df = trainer.load_and_prepare_data()

            history = trainer.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=32)

            metrics, predictions, y_true = trainer.evaluate(X_test, y_test)

            trainer.save_artifacts()

            results[coin] = {
                'success': True,
                'metrics': metrics
            }

            print(f"\n✓ Successfully trained model for {coin}")

        except Exception as e:
            print(f"\n✗ Error training model for {coin}: {e}")
            results[coin] = {
                'success': False,
                'error': str(e)
            }

    print(f"\n\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for coin, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{coin.upper()}: {status}")
        if result['success']:
            print(f"  MAPE: {result['metrics']['MAPE']:.2f}%")
            print(f"  RMSE: {result['metrics']['RMSE']:.4f}")

    return results


if __name__ == "__main__":
    print("CryptoBud - Model Training Pipeline")
    print("="*60)

    train_single = True

    if train_single:
        coin = 'bitcoin'
        trainer = ModelTrainer(coin, sequence_length=90, prediction_steps=5)

        X_train, X_val, X_test, y_train, y_val, y_test, df = trainer.load_and_prepare_data()

        history = trainer.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

        metrics, predictions, y_true = trainer.evaluate(X_test, y_test)

        trainer.save_artifacts()
    else:
        results = train_all_coins(coins=['bitcoin', 'ethereum'], epochs=50)
