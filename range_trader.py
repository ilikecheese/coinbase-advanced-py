from backtest import run_backtest_with_risk_management
from data_loader import load_candle_data
import pandas as pd  # Import pandas for date handling
from strategy import find_optimal_trades, find_optimal_strategy_trades
from atr_threshold import update_strategy_with_dynamic_threshold

# Coinbase Advanced Fee Structure (as of May 2025):
# $0-$1K monthly: Maker 0.60%, Taker 1.20%
# $1K-$500K monthly: Maker 0.35%, Taker 0.75%
# For this backtest, default to Maker 0.60% (limit order) and Taker 1.20% (market order)
# Set fee_pct to 0.60 for Maker, 1.20 for Taker. Adjust as needed for your scenario.

PARAMETERS = {
    "csv_file": "market_data/AMP_USDC/master_data.csv",  # Change as needed
    "pair_name": "AMP/USDC",
    "threshold_pct": 5, # Percentage threshold for buy/sell signals (ignored if ATR is enabled)
    "initial_cash": 100,    
    "initial_crypto": 100,  # Amount in USD to be converted to crypto units at start (e.g., 100 means $100 worth of crypto)
    "min_cash_reserve": 20,  # Minimum cash reserve to keep in the portfolio
    # Choose one of the following: #
    # "fee_pct": 0.60,  # Maker (limit order)
    "fee_pct": 1.20,   # Taker (market order)
    "backtest_days": 30,   # Number of days to backtest ("None" = all data)
    "stop_loss_pct": None,  # 2% stop-loss (set to None to disable)
    "take_profit_pct": None, # 3% take-profit (set to None to disable)
    # ATR dynamic threshold settings (uncomment to enable)
     "atr_period": 14,  # ATR period (set to None to disable)
     "atr_k": 1.5,      # ATR scaling factor (set to None to disable)
}

def main():
    df = load_candle_data(PARAMETERS["csv_file"])
    df["date"] = pd.to_datetime(df["date"])
    if PARAMETERS.get("backtest_days") is not None:
        cutoff = df["date"].max() - pd.Timedelta(days=PARAMETERS["backtest_days"])
        df = df[df["date"] >= cutoff]
    print(f"Backtest date range: {df['date'].min()} to {df['date'].max()} ({len(df)} rows)")
    start_price = df.iloc[0]["close"]
    initial_crypto_amount = PARAMETERS["initial_crypto"] / start_price
    # Convert string 'None' to Python None for stop_loss_pct/take_profit_pct
    params = {**PARAMETERS, "initial_crypto": initial_crypto_amount}
    if params.get('stop_loss_pct') == 'None':
        params['stop_loss_pct'] = None
    if params.get('take_profit_pct') == 'None':
        params['take_profit_pct'] = None
    # Integrate ATR-based dynamic thresholds if enabled
    if params.get('atr_period') is not None and params.get('atr_k') is not None:
        df = update_strategy_with_dynamic_threshold(df, params)
    trade_log, portfolio_value = run_backtest_with_risk_management(df, params)
    # Calculate final portfolio value in USD (cash + crypto * last close price)
    final_price = df.iloc[-1]["close"]
    # Defensive: handle empty trade_log
    if len(trade_log) == 0:
        print("No trades executed. Portfolio remains in initial state.")
        final_portfolio_value_usd = params["initial_cash"] + initial_crypto_amount * final_price
        print(f"Final portfolio value: {final_portfolio_value_usd:.2f} USD (USDC: ${params['initial_cash']:.2f}, {params['pair_name'].split('/')[0]}: ${initial_crypto_amount * final_price:.2f}, {params['pair_name'].split('/')[0]} qty: {initial_crypto_amount:.6f})")
        return

    final_cash = trade_log[-1].get('cash', None) if 'cash' in trade_log[-1] else None
    final_crypto = trade_log[-1].get('crypto', None) if 'crypto' in trade_log[-1] else None
    if final_cash is not None and final_crypto is not None:
        final_portfolio_value_usd = final_cash + final_crypto * final_price
        print(f"Final portfolio value: {final_portfolio_value_usd:.2f} USD (USDC: ${final_cash:.2f}, {params['pair_name'].split('/')[0]}: ${final_crypto * final_price:.2f}, {params['pair_name'].split('/')[0]} qty: {final_crypto:.6f})")
    else:
        # Fallback to the old calculation if cash/crypto not tracked in trade_log
        final_portfolio_value_usd = params["initial_cash"] + initial_crypto_amount * final_price
        print(f"Final portfolio value: {final_portfolio_value_usd:.2f} USD (USDC: ${params['initial_cash']:.2f}, {params['pair_name'].split('/')[0]}: ${initial_crypto_amount * final_price:.2f}, {params['pair_name'].split('/')[0]} qty: {initial_crypto_amount:.6f})")
    print(f"Trade log (first 5 shown):")
    # Print trade log in columns
    header = f"{'Type':<8} {'Entry':>10} {'Exit':>10} {'Qty':>12} {'Fee':>10} {'Date':>20} {'SL':>10} {'TP':>10} {'Reason':>10}"
    print(header)
    print('-' * len(header))
    for trade in trade_log[:5]:
        action = trade.get('action', '')
        entry_val = trade.get('entry_price')
        exit_val = trade.get('exit_price')
        # Show crypto as USD value (qty * price at trade)
        qty_val = trade.get('qty')
        # Use entry price for buys, exit price for sells/exits
        if action == 'buy' and entry_val not in (None, '') and qty_val not in (None, ''):
            qty_usd = qty_val * entry_val
        elif action in ('sell', 'exit') and exit_val not in (None, '') and qty_val not in (None, ''):
            qty_usd = qty_val * exit_val
        else:
            qty_usd = ''
        entry = f"{entry_val:.5f}" if entry_val not in (None, '') else ''
        exit_ = f"{exit_val:.5f}" if exit_val not in (None, '') else ''
        qty = f"{qty_usd:,.2f}" if qty_usd not in (None, '') else ''
        fee_val = trade.get('fee')
        fee = f"{fee_val:.4f}" if fee_val not in (None, '') else ''
        date = str(trade.get('date', ''))[:19]
        sl_val = trade.get('stop_loss_price')
        sl = f"{sl_val:.5f}" if sl_val not in (None, '') else ''
        tp_val = trade.get('take_profit_price')
        tp = f"{tp_val:.5f}" if tp_val not in (None, '') else ''
        reason = trade.get('exit_reason', '')
        print(f"{action:<8} {entry:>10} {exit_:>10} {qty:>12} {fee:>10} {date:>20} {sl:>10} {tp:>10} {reason:>10}")
    print(f"Total trades: {len(trade_log)}")
    # --- Optimal vs Probable Trade Comparison ---
    print("\nOptimal vs. Probable Trade Comparison:")
    optimal_trades = find_optimal_strategy_trades(
        df,
        PARAMETERS["threshold_pct"],
        PARAMETERS["initial_cash"],
        PARAMETERS["initial_crypto"],
        PARAMETERS["min_cash_reserve"],
        PARAMETERS["fee_pct"]
    )
    print(f"Optimal trades: {len(optimal_trades)} (strategy-constrained)")
    print(f"Probable trades: {len(trade_log)} (your strategy)")
    print("First 5 optimal trades:")
    for t in optimal_trades[:5]:
        print(f"{t[0].title()} @ {t[1]:.5f} on {t[4]}")
    # Re-run plot with optimal trades overlay (optional, can be updated to use new plot if desired)
    # run_backtest(df, {**PARAMETERS, "initial_crypto": initial_crypto_amount, "optimal_trades": optimal_trades})

if __name__ == "__main__":
    main()
