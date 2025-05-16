from backtest import run_backtest
from data_loader import load_candle_data
from visualization import plot_price_with_trades

PARAMETERS = {
    "csv_file": "market_data/AMP_USDC/master_data.csv",  # Change as needed
    "pair_name": "AMP/USDC",
    "threshold_pct": 5,
    "initial_cash": 100,
    "initial_crypto": 100,
    "min_cash_reserve": 20,
    "fee_pct": 0.1,
}

def main():
    df = load_candle_data(PARAMETERS["csv_file"])
    start_price = df.iloc[0]["close"]
    # Convert initial_crypto (USD value) to crypto amount at start price
    initial_crypto_amount = PARAMETERS["initial_crypto"] / start_price
    # Pass the crypto amount to the backtest, not Portfolio directly
    result = run_backtest({**PARAMETERS, "initial_crypto": initial_crypto_amount})
    print(f"Final portfolio value: {result.value(df.iloc[-1]['close']):.2f} (cash: {result.cash:.2f} USD, crypto: {result.crypto * df.iloc[-1]['close']:.2f} USD, crypto amount: {result.crypto:.6f})")
    print("Trade history:")
    for trade in result.history:
        print(trade)
    # Visualization with zone highlighting
    plot_price_with_trades(df, result.history, PARAMETERS["pair_name"], threshold_pct=PARAMETERS["threshold_pct"])

if __name__ == "__main__":
    main()
