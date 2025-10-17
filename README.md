# CryptoBud - Real-Time Cryptocurrency Dashboard with AI Predictions

A real-time cryptocurrency dashboard built with Python that displays live price data and LSTM-powered predictions for major cryptocurrencies.

## Features

- **Real-time price tracking** for Bitcoin, Ethereum, XRP, and other major cryptocurrencies
- **LSTM neural network predictions** showing future price trends
- **Interactive dashboard** with live updating charts
- **Prediction accuracy metrics** showing delta between predictions and actual prices
- **Multiple cryptocurrency support** with easy coin selection

## Tech Stack

- **Frontend**: Streamlit + Plotly
- **ML Model**: TensorFlow/Keras LSTM
- **Data Source**: CoinGecko API / Binance WebSocket
- **Deployment**: Streamlit Cloud / Railway

## Installation

### 1. Create Conda Environment

```bash
conda create -n cryptobud python=3.11
conda activate cryptobud
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Usage

### Train Models

```bash
python models/train_model.py
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
cryptobud/
├── data/
│   ├── fetch_crypto_data.py      # Real-time data fetching
│   └── historical_data.py         # Historical data collection
├── models/
│   ├── lstm_model.py              # LSTM architecture
│   ├── train_model.py             # Training pipeline
│   └── predict.py                 # Real-time predictions
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── utils/
│   └── preprocessing.py           # Data preprocessing
├── saved_models/                  # Trained model files
├── requirements.txt
└── README.md
```

## How It Works

1. **Data Collection**: Fetches real-time cryptocurrency prices via CoinGecko API or Binance WebSocket
2. **Preprocessing**: Normalizes data using MinMaxScaler and creates sequences for LSTM
3. **Prediction**: LSTM model trained on historical data predicts 5-10 steps ahead
4. **Visualization**: Streamlit dashboard shows actual prices (solid line) and predictions (dashed line)
5. **Metrics**: Displays prediction accuracy with MAPE and RMSE metrics

## Model Architecture

- 3 LSTM layers with 50-100 units each
- 20% dropout for regularization
- Dense output layer
- Trained on 90 timesteps to predict next 5-10 steps
- Adam optimizer with MSE loss

## Deployment

Deploy to Streamlit Cloud for free hosting:

```bash
streamlit cloud deploy
```

## Contributing

Feel free to open issues or submit pull requests!

## License

MIT License

## Author

Brian Ellis
