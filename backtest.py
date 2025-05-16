from data_loader import load_candle_data
from portfolio import Portfolio
from strategy import should_buy, should_sell
from visualization import plot_portfolio_value_over_time, plot_price_and_portfolio, plot_price_with_trades_and_portfolio

import logging

def run_backtest(params):
    df = load_candle_data(params["csv_file"])
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
    # Plot portfolio value over time and save to file
    portfolio_history = portfolio.value_history(df)
    plot_portfolio_value_over_time(
        portfolio_history,
        pair_name=params.get("pair_name", "Crypto/USDC"),
        save_path="plots/portfolio_value_test.png",
        show=True
    )
    # Plot combined price and portfolio value chart
    plot_price_and_portfolio(
        df,
        portfolio_history,
        pair_name=params.get("pair_name", "Crypto/USDC"),
        save_path="plots/price_and_portfolio_test.png",
        show=True
    )
    # Plot price with buy/sell trades and portfolio value at each trade
    plot_price_with_trades_and_portfolio(
        df,
        portfolio.history,
        portfolio_history,
        pair_name=params.get("pair_name", "Crypto/USDC"),
        threshold_pct=params.get("threshold_pct", None)
    )
    return portfolio
