import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

class DataPreprocessor:
    def __init__(self, sequence_length=90):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, df, target_column='price', feature_columns=None):
        if feature_columns is None:
            feature_columns = ['price']

        df = df.sort_values('timestamp').reset_index(drop=True)

        df = df.dropna(subset=feature_columns)

        target_data = df[target_column].values.reshape(-1, 1)
        self.scaler.fit(target_data)
        scaled_target = self.scaler.transform(target_data)

        if len(feature_columns) > 1:
            feature_data = df[feature_columns].values
            self.feature_scaler.fit(feature_data)
            scaled_features = self.feature_scaler.transform(feature_data)
        else:
            scaled_features = scaled_target

        return scaled_features, df

    def create_sequences(self, data, prediction_steps=5):
        X, y = [], []

        for i in range(self.sequence_length, len(data) - prediction_steps + 1):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i:i + prediction_steps, 0])

        return np.array(X), np.array(y)

    def train_test_split(self, X, y, train_size=0.8):
        split_idx = int(len(X) * train_size)

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        return X_train, X_test, y_train, y_test

    def inverse_transform_predictions(self, predictions):
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return self.scaler.inverse_transform(predictions)

    def prepare_for_prediction(self, recent_data):
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")

        recent_data = recent_data[-self.sequence_length:]
        recent_data = recent_data.reshape(-1, 1)

        scaled_data = self.scaler.transform(recent_data)

        X = scaled_data.reshape(1, self.sequence_length, -1)

        return X

    def save_scaler(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'feature_scaler': self.feature_scaler}, f)
        print(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath):
        with open(filepath, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler = scalers['scaler']
            self.feature_scaler = scalers['feature_scaler']
        print(f"Scaler loaded from {filepath}")


class TimeSeriesValidator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

    @staticmethod
    def directional_accuracy(y_true, y_pred):
        y_true_direction = np.diff(y_true.flatten()) > 0
        y_pred_direction = np.diff(y_pred.flatten()) > 0

        accuracy = np.mean(y_true_direction == y_pred_direction) * 100

        return accuracy


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    prices = np.cumsum(np.random.randn(200)) + 100

    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.rand(200) * 1000000
    })

    print("Sample data:")
    print(df.head())

    preprocessor = DataPreprocessor(sequence_length=90)

    scaled_data, processed_df = preprocessor.prepare_data(df, target_column='price')
    print(f"\nScaled data shape: {scaled_data.shape}")

    X, y = preprocessor.create_sequences(scaled_data, prediction_steps=5)
    print(f"Sequences shape - X: {X.shape}, y: {y.shape}")

    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y, train_size=0.8)
    print(f"\nTrain set - X: {X_train.shape}, y: {y_train.shape}")
    print(f"Test set - X: {X_test.shape}, y: {y_test.shape}")

    sample_predictions = y_test[0]
    inverse_pred = preprocessor.inverse_transform_predictions(sample_predictions)
    print(f"\nSample prediction (scaled): {sample_predictions[:3]}")
    print(f"Sample prediction (original scale): {inverse_pred[:3]}")
