from portfolio import Portfolio
from strategy import should_buy, should_sell, run_strategy_with_risk_management
from visualization import plot_price_with_trades, plot_price_with_trade_log
import os
import logging

def run_backtest(df, params):
    portfolio = Portfolio(params["initial_cash"], params["initial_crypto"], params["fee_pct"])
    threshold_pct = params["threshold_pct"]
    min_cash_reserve = params["min_cash_reserve"]
    last_trade_price = df.iloc[0]["close"]

    for idx, row in df.iterrows():
        price = row["close"]
        # Buy
        if portfolio.cash > min_cash_reserve and should_buy(last_trade_price, price, threshold_pct):
            portfolio.buy(price, threshold_pct / 100)
            last_trade_price = price
        # Sell
        elif portfolio.crypto > 0 and should_sell(last_trade_price, price, threshold_pct):
            portfolio.sell(price, threshold_pct / 100)
            last_trade_price = price
    # Save plot in plots folder
    os.makedirs("plots", exist_ok=True)
    start_date = df["date"].iloc[0].strftime("%Y-%m-%d")
    end_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    pair = params.get("pair_name", "Crypto_USDC").replace("/", "_")
    save_path = f"plots/{pair}_{start_date}_to_{end_date}.png"
    # Pass optimal_trades if present in params
    optimal_trades = params.get("optimal_trades", None)
    plot_price_with_trades(
        df,
        portfolio.history,
        params.get("pair_name", "Crypto/USDC"),
        threshold_pct=params.get("threshold_pct", None),
        save_path=save_path,
        optimal_trades=optimal_trades
    )
    return portfolio

def run_backtest_with_risk_management(df, params):
    """
    Run backtest using strategy with stop-loss and take-profit logic.
    Returns trade log and summary portfolio info.
    """
    # Ensure stop_loss_pct and take_profit_pct are in params
    params.setdefault('stop_loss_pct', 0.02)  # Default 2%
    params.setdefault('take_profit_pct', 0.03)  # Default 3%
    trade_log = run_strategy_with_risk_management(df, params)
    # Optionally, compute final portfolio value
    final_price = df.iloc[-1]['close']
    cash = params['initial_cash']
    crypto = params['initial_crypto'] / df.iloc[0]['close'] if params['initial_crypto'] > 0 else 0
    for trade in trade_log:
        if trade['action'] == 'buy':
            cash -= trade['entry_price'] * trade['qty'] + trade['fee']
            crypto += trade['qty']
        elif trade['action'] in ('sell', 'exit'):
            cash += trade['exit_price'] * trade['qty'] - trade['fee']
            crypto -= trade['qty']
    portfolio_value = cash + crypto * final_price
    # Remove or comment out any print statements here to avoid duplicate/confusing output
    # print(f"Final portfolio value: {portfolio_value:.2f} (cash: {cash:.2f}, crypto: {crypto:.6f})")
    # Visualization: plot with stop-loss/take-profit markers
    os.makedirs("plots", exist_ok=True)
    start_date = df["date"].iloc[0].strftime("%Y-%m-%d")
    end_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    pair = params.get("pair_name", "Crypto_USDC").replace("/", "_")
    save_path = f"plots/{pair}_{start_date}_to_{end_date}_risk.png"
    plot_price_with_trade_log(df, trade_log, pair_name=params.get("pair_name", "Crypto/USDC"), save_path=save_path)
    return trade_log, portfolio_value
