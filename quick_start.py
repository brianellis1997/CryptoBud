import os
import sys

def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        CryptoBud - Quick Start Setup                      ║
    ║        Real-Time Crypto Dashboard with AI Predictions    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    print("\n[1/4] Checking dependencies...")

    required_modules = [
        'streamlit', 'pandas', 'numpy', 'plotly',
        'tensorflow', 'sklearn', 'requests', 'pycoingecko'
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - MISSING")
            missing.append(module)

    if missing:
        print("\n⚠ Missing dependencies. Install with:")
        print("  python -m pip install -r requirements.txt")
        return False

    print("\n✓ All dependencies installed!")
    return True

def fetch_data():
    print("\n[2/4] Fetching historical data...")

    try:
        from data.historical_data import HistoricalDataFetcher

        fetcher = HistoricalDataFetcher()
        df = fetcher.fetch_all_coins(days=180)

        if not df.empty:
            df_with_indicators = fetcher.add_technical_indicators(df)

            os.makedirs('data', exist_ok=True)
            filename = 'data/historical_crypto_data.csv'
            fetcher.save_data(df_with_indicators, filename)

            print(f"✓ Fetched {len(df)} data points")
            print(f"✓ Saved to {filename}")
            return True
        else:
            print("✗ No data fetched")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def train_models(quick_mode=True):
    print("\n[3/4] Training LSTM models...")
    print("  (This may take several minutes...)")

    try:
        from models.train_model import ModelTrainer

        coins = ['bitcoin', 'ethereum'] if quick_mode else ['bitcoin', 'ethereum', 'ripple']
        epochs = 30 if quick_mode else 50

        for coin in coins:
            print(f"\n  Training model for {coin}...")

            trainer = ModelTrainer(coin, sequence_length=90, prediction_steps=5)

            X_train, X_val, X_test, y_train, y_val, y_test, df = trainer.load_and_prepare_data(
                data_path='data/historical_crypto_data.csv'
            )

            history = trainer.train(
                X_train, y_train, X_val, y_val,
                epochs=epochs, batch_size=32
            )

            metrics, predictions, y_true = trainer.evaluate(X_test, y_test)

            trainer.save_artifacts()

            print(f"  ✓ {coin} - MAPE: {metrics['MAPE']:.2f}%")

        print("\n✓ All models trained successfully!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def launch_dashboard():
    print("\n[4/4] Launching dashboard...")
    print("\n" + "="*60)
    print("Dashboard starting at http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    os.system("streamlit run dashboard/app.py")

def main():
    print_banner()

    print("\nCryptoBud Quick Start")
    print("-" * 60)
    print("This script will:")
    print("  1. Check dependencies")
    print("  2. Fetch historical crypto data")
    print("  3. Train LSTM models")
    print("  4. Launch the dashboard")
    print("-" * 60)

    choice = input("\nContinue? (y/n): ").lower()
    if choice != 'y':
        print("Setup cancelled.")
        return

    quick_mode = True
    if input("\nQuick mode (train only BTC & ETH)? (y/n): ").lower() != 'y':
        quick_mode = False

    if not check_requirements():
        return

    if not fetch_data():
        print("\n⚠ Data fetch failed. Cannot proceed.")
        return

    if not train_models(quick_mode=quick_mode):
        print("\n⚠ Model training failed. Cannot proceed.")
        return

    print("\n" + "="*60)
    print("✓ Setup complete!")
    print("="*60)

    if input("\nLaunch dashboard now? (y/n): ").lower() == 'y':
        launch_dashboard()
    else:
        print("\nTo launch the dashboard later, run:")
        print("  streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()
