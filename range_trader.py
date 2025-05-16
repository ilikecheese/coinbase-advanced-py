from backtest import run_backtest
from data_loader import load_candle_data

# Coinbase Advanced Fee Structure (as of May 2025):
# $0-$1K monthly: Maker 0.60%, Taker 1.20%
# $1K-$500K monthly: Maker 0.35%, Taker 0.75%
# For this backtest, default to Maker 0.35% (limit order) and Taker 0.75% (market order)
# Set fee_pct to 0.35 for Maker, 0.75 for Taker. Adjust as needed for your scenario.

PARAMETERS = {
    "csv_file": "market_data/AMP_USDC/master_data.csv",  # Change as needed
    "pair_name": "AMP/USDC",
    "threshold_pct": 5,
    "initial_cash": 100,
    "initial_crypto": 100,
    "min_cash_reserve": 20,
    # Choose one of the following:
    # "fee_pct": 0.35,  # Maker (limit order)
    "fee_pct": 0.75,   # Taker (market order)
}

def main():
    df = load_candle_data(PARAMETERS["csv_file"])
    start_price = df.iloc[0]["close"]
    # Convert initial_crypto (USD value) to crypto amount at start price
    initial_crypto_amount = PARAMETERS["initial_crypto"] / start_price
    # Pass the crypto amount to the backtest, not Portfolio directly
    result = run_backtest({**PARAMETERS, "initial_crypto": initial_crypto_amount})
    print(f"Final portfolio value: {result.value(df.iloc[-1]['close']):.2f} (cash: {result.cash:.2f} USD, crypto: {result.crypto * df.iloc[-1]['close']:.2f} USD, crypto qty: {result.crypto:.6f})")
    print(f"Trade history for {PARAMETERS['pair_name']}:")
    print(f"{'Type':<6} {'Crypto Price':>10} {'USD Value':>12} {'Qty':>14} {'Fee (USD)':>12} {'Crypto Balance':>16} {'USDC Balance':>18}")
    # Track running portfolio value after each trade
    cash = PARAMETERS["initial_cash"]
    crypto = initial_crypto_amount
    fee_pct = PARAMETERS["fee_pct"]
    for trade in result.history:
        action, price, qty, fee = trade
        usd_value = qty * price
        if action == "buy":
            cash -= usd_value + fee
            crypto += qty
        elif action == "sell":
            cash += usd_value - fee
            crypto -= qty
        crypto_value = crypto * price
        portfolio_value = cash + crypto_value
        print(f"{action.title():<6} {price:>10.5f} {usd_value:>12.2f} {qty:>14.6f} {fee:>12.4f} {crypto_value:>16.2f} {portfolio_value:>18.2f}")
    # Visualization is handled in run_backtest; do not call plot_price_with_trades here.

if __name__ == "__main__":
    main()
