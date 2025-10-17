# CryptoBud - Real-Time Cryptocurrency Dashboard with AI Predictions

A real-time cryptocurrency dashboard built with Python that displays live price data and LSTM-powered predictions for major cryptocurrencies.

## Features

- **Real-time price tracking** for Bitcoin, Ethereum, XRP, and other major cryptocurrencies
- **WebSocket streaming** - Updates every second via Binance WebSocket (no API rate limits!)
- **Demo mode available** - View live prices without training any models
- **LSTM neural network predictions** showing future price trends (5 steps ahead)
- **Interactive dashboard** with live updating charts using Plotly
- **Prediction accuracy metrics** showing delta between predictions and actual prices
- **Multiple cryptocurrency support** with easy coin selection
- **Technical indicators** - SMA, EMA, RSI, volatility calculations

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

### Quick Start - Demo Mode (No Training Required!)

Try the real-time dashboard immediately:

```bash
conda activate cryptobud
streamlit run dashboard/demo_app.py
```

**Features:**
- Real-time prices updated every second
- Live charts with price history
- No model training needed
- Perfect for testing and exploration

### Full Mode with AI Predictions

#### Option 1: Automated Setup (Recommended)

```bash
python quick_start.py
```

This will fetch data, train models, and launch the dashboard.

#### Option 2: Manual Setup

```bash
# 1. Fetch historical data
python data/historical_data.py

# 2. Train models (takes 25-50 minutes for all coins)
python models/train_model.py

# 3. Run dashboard with predictions
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
cryptobud/
├── data/
│   ├── fetch_crypto_data.py      # CoinGecko API fetching
│   ├── binance_websocket.py      # Real-time WebSocket streaming
│   └── historical_data.py         # Historical data collection
├── models/
│   ├── lstm_model.py              # LSTM architecture
│   ├── train_model.py             # Training pipeline
│   └── predict.py                 # Real-time predictions
├── dashboard/
│   ├── app.py                     # Full dashboard with predictions
│   └── demo_app.py                # Demo mode (no training needed)
├── utils/
│   └── preprocessing.py           # Data preprocessing
├── saved_models/                  # Trained model files
├── requirements.txt
├── README.md
├── USAGE.md                       # Detailed usage guide
├── DEPLOYMENT.md                  # Deployment instructions
├── QUICK_ANSWERS.md               # FAQ and quick reference
└── NEXT_STEPS.md                  # Enhancement ideas
```

## How It Works

### Demo Mode (Real-Time Prices Only)
1. **WebSocket Connection**: Connects to Binance WebSocket for live price streaming
2. **Real-Time Updates**: Receives price updates every second (no API limits)
3. **Live Visualization**: Updates charts in real-time with price history
4. **Multi-Coin View**: Monitor multiple cryptocurrencies simultaneously

### Full Mode (With AI Predictions)
1. **Data Collection**: Fetches historical data via CoinGecko API for training
2. **Technical Indicators**: Calculates SMA, EMA, RSI, volatility on historical data
3. **Preprocessing**: Normalizes data using MinMaxScaler and creates 90-step sequences
4. **LSTM Training**: Trains 3-layer LSTM model (100/75/50 units) with 20% dropout
5. **Prediction**: Model predicts next 5 price points based on last 90 data points
6. **Visualization**: Dashboard shows actual prices (solid) vs predictions (dashed)
7. **Metrics**: Displays MAPE, RMSE, MAE, and directional accuracy

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
