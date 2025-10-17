# CryptoBud Usage Guide

Complete guide on how to use CryptoBud for cryptocurrency price prediction and visualization.

## Quick Start

### Method 1: Automated Setup (Recommended)

Run the quick start script:
```bash
conda activate cryptobud
python quick_start.py
```

This will:
1. Check dependencies
2. Fetch historical data
3. Train models
4. Launch the dashboard

### Method 2: Manual Setup

#### Step 1: Fetch Historical Data

```bash
conda activate cryptobud
cd data
python historical_data.py
```

This downloads 180 days of historical price data for all supported cryptocurrencies.

#### Step 2: Train Models

Train models for all coins:
```bash
cd models
python train_model.py
```

Or train for specific coins by editing the script.

#### Step 3: Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

## Using the Dashboard

### Main Features

#### 1. **Cryptocurrency Selection**
- Use the sidebar dropdown to select a cryptocurrency
- Available coins: BTC, ETH, XRP, ADA, SOL

#### 2. **Timeframe Selection**
- Choose historical data timeframe: 1, 3, 7, 14, or 30 days
- Affects the chart's historical data display

#### 3. **Real-Time Price Display**
- Current price with 24h change percentage
- Market cap
- 24h trading volume
- Last update timestamp

#### 4. **Price Chart**
- Blue line: Actual historical prices
- Red dashed line: AI predictions (when enabled)
- Volume bar chart below price chart
- Hover for detailed information

#### 5. **Predictions Panel**
- Shows next 5 price predictions
- Displays percentage change for each prediction
- Color-coded: Green (bullish) / Red (bearish)
- Overall trend indicator

#### 6. **Delta Analysis**
- Bar chart showing predicted price changes
- Helps visualize bullish/bearish momentum
- Percentage change from current price

### Dashboard Controls

#### Auto Refresh
Enable to automatically refresh data every 60 seconds:
- Check "Auto Refresh" in sidebar
- Useful for monitoring live price changes

#### Manual Refresh
Click the "🔄 Refresh Data" button to update:
- Current prices
- Charts
- Predictions

#### Show/Hide Predictions
Toggle the "Show Predictions" checkbox to:
- Display or hide prediction overlay
- Useful for comparing actual vs predicted prices

## Understanding the Predictions

### How It Works

1. **Data Collection**: Fetches last 90 data points (prices)
2. **Preprocessing**: Normalizes data using MinMaxScaler
3. **Prediction**: LSTM model predicts next 5 time steps
4. **Post-processing**: Converts predictions back to actual prices

### Prediction Steps

Each step represents a future time period:
- **Step 1**: Next time period (e.g., +1 hour)
- **Step 2**: +2 time periods
- **Step 3**: +3 time periods
- **Step 4**: +4 time periods
- **Step 5**: +5 time periods

### Interpreting Results

#### Price Change Percentage
- **Positive %**: Predicted price increase (bullish)
- **Negative %**: Predicted price decrease (bearish)

#### Trend Indicator
- **BULLISH 📈**: Average predicted change is positive
- **BEARISH 📉**: Average predicted change is negative

#### Confidence
- Predictions closer to current time (Step 1) are generally more reliable
- Further predictions (Step 5) have higher uncertainty

## Advanced Usage

### Custom Data Fetching

Fetch data for specific coins:
```python
from data.fetch_crypto_data import CryptoDataFetcher

fetcher = CryptoDataFetcher()

# Get current prices
df = fetcher.get_current_prices(['bitcoin', 'ethereum'])
print(df)

# Get minute-level data
btc_data = fetcher.get_minute_prices('bitcoin', days=7)
print(btc_data)
```

### Training Custom Models

Train with custom parameters:
```python
from models.train_model import ModelTrainer

trainer = ModelTrainer('bitcoin', sequence_length=120, prediction_steps=10)

X_train, X_val, X_test, y_train, y_val, y_test, df = trainer.load_and_prepare_data()

history = trainer.train(
    X_train, y_train, X_val, y_val,
    epochs=100,
    batch_size=64
)

metrics, predictions, y_true = trainer.evaluate(X_test, y_test)
```

### Making Predictions

Use the prediction module directly:
```python
from models.predict import CryptoPricePredictor

predictor = CryptoPricePredictor('bitcoin')
predictor.load_model_and_scaler()

prediction_info = predictor.predict_realtime()

print(f"Current Price: ${prediction_info['current_price']:.2f}")
print("Predictions:", prediction_info['predictions'])
```

### Multi-Coin Predictions

Get predictions for multiple coins:
```python
from models.predict import MultiCoinPredictor

coins = ['bitcoin', 'ethereum', 'ripple']
multi_predictor = MultiCoinPredictor(coins)

predictions = multi_predictor.predict_all()

summary = multi_predictor.get_summary(predictions)
print(summary)
```

## Data Management

### Viewing Historical Data

```python
from data.historical_data import HistoricalDataFetcher

fetcher = HistoricalDataFetcher()
df = fetcher.load_data('data/historical_crypto_data.csv')

print(df[df['coin_id'] == 'bitcoin'].tail())
```

### Adding Technical Indicators

```python
df_with_indicators = fetcher.add_technical_indicators(df)

print(df_with_indicators.columns)
```

Available indicators:
- Returns (percentage change)
- SMA (7-day, 30-day)
- EMA (12-day, 26-day)
- RSI (14-day)
- Volatility (7-day)

## Model Management

### Saved Model Files

Models are saved in `saved_models/` directory:
- `{coin}_lstm.keras` - Trained model
- `{coin}_scaler.pkl` - Data scaler
- `{coin}_metrics.json` - Performance metrics
- `{coin}_training_history.png` - Training plots

### Loading Pre-trained Models

```python
from models.lstm_model import CryptoLSTMModel

model = CryptoLSTMModel(sequence_length=90, n_features=1, prediction_steps=5)
model.load_model('saved_models/bitcoin_lstm.keras')
```

### Evaluating Model Performance

Check metrics file:
```python
import json

with open('saved_models/bitcoin_metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"RMSE: {metrics['RMSE']:.4f}")
```

## Tips & Best Practices

### For Accurate Predictions

1. **Train on sufficient data**: Use at least 180 days of historical data
2. **Retrain regularly**: Market conditions change; retrain models weekly
3. **Monitor performance**: Check MAPE and RMSE metrics
4. **Consider multiple models**: Different models may work better for different coins

### For Better Performance

1. **Use caching**: Dashboard already implements Streamlit caching
2. **Batch predictions**: Predict multiple coins at once
3. **Optimize data fetching**: Cache API responses when possible
4. **Pre-load models**: Load models once and reuse

### Risk Management

**Important**: These predictions are for educational purposes only!

- Don't base trading decisions solely on predictions
- Cryptocurrency markets are highly volatile
- Past performance doesn't guarantee future results
- Always do your own research (DYOR)

## Troubleshooting

### "Model not found" error
```bash
# Train the model first
python models/train_model.py
```

### "Not enough data" error
```bash
# Fetch fresh historical data
python data/historical_data.py
```

### Slow dashboard performance
- Reduce auto-refresh frequency
- Use shorter timeframes
- Close other resource-intensive applications

### Prediction accuracy issues
- Retrain with more data
- Adjust model hyperparameters
- Try different sequence lengths

## Command Reference

### Data Commands
```bash
# Fetch current prices
python -c "from data.fetch_crypto_data import CryptoDataFetcher; CryptoDataFetcher().get_current_prices()"

# Fetch historical data
python data/historical_data.py
```

### Model Commands
```bash
# Train all models
python models/train_model.py

# Test predictions
python models/predict.py
```

### Dashboard Commands
```bash
# Launch dashboard
streamlit run dashboard/app.py

# Launch on specific port
streamlit run dashboard/app.py --server.port 8502
```

## API Rate Limits

### CoinGecko API
- Free tier: 10-50 calls/minute
- Upgrade for higher limits
- Dashboard implements rate limiting

### Best Practices
- Cache frequently accessed data
- Use batch API calls when possible
- Respect rate limits

## Getting Help

### Resources
- [Streamlit Documentation](https://docs.streamlit.io)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide/keras)
- [CoinGecko API Docs](https://www.coingecko.com/en/api/documentation)

### Support
- Open an issue on GitHub
- Check existing issues for solutions
- Review error logs for debugging

## Next Steps

1. **Experiment with parameters**: Try different sequence lengths, prediction steps
2. **Add more coins**: Extend to support additional cryptocurrencies
3. **Enhance features**: Add alerts, portfolio tracking, backtesting
4. **Deploy**: Share your dashboard with the world!

Happy predicting! 🚀
