import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

class CryptoLSTMModel:
    def __init__(self, sequence_length=90, n_features=1, prediction_steps=5):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.prediction_steps = prediction_steps
        self.model = None
        self.history = None

    def build_model(self, lstm_units=[100, 75, 50], dropout_rate=0.2, learning_rate=0.001):
        self.model = Sequential()

        self.model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.n_features)
        ))
        self.model.add(Dropout(dropout_rate))

        for units in lstm_units[1:]:
            self.model.add(LSTM(units=units, return_sequences=True))
            self.model.add(Dropout(dropout_rate))

        self.model.add(LSTM(units=lstm_units[-1], return_sequences=False))
        self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(units=50, activation='relu'))
        self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(units=self.prediction_steps))

        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mape']
        )

        return self.model

    def get_callbacks(self, model_path, patience=15):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]

        return callbacks

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_path='model.keras'):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        callbacks = self.get_callbacks(model_path, patience=15)

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        predictions = self.model.predict(X, verbose=0)
        return predictions

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))

        return metrics

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save.")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

        return self.model

    def summary(self):
        if self.model is None:
            raise ValueError("Model not built.")

        return self.model.summary()


if __name__ == "__main__":
    print("Testing LSTM Model...")

    sequence_length = 90
    n_features = 1
    prediction_steps = 5
    n_samples = 1000

    X_train = np.random.randn(n_samples, sequence_length, n_features)
    y_train = np.random.randn(n_samples, prediction_steps)
    X_val = np.random.randn(200, sequence_length, n_features)
    y_val = np.random.randn(200, prediction_steps)

    print("\nCreating LSTM model...")
    lstm_model = CryptoLSTMModel(
        sequence_length=sequence_length,
        n_features=n_features,
        prediction_steps=prediction_steps
    )

    model = lstm_model.build_model(
        lstm_units=[100, 75, 50],
        dropout_rate=0.2,
        learning_rate=0.001
    )

    print("\nModel Summary:")
    lstm_model.summary()

    print("\nModel built successfully!")
    print(f"Input shape: (batch_size, {sequence_length}, {n_features})")
    print(f"Output shape: (batch_size, {prediction_steps})")
