from data_loader import load_candle_data
from portfolio import Portfolio
from strategy import should_buy, should_sell
from visualization import plot_price_with_trades

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
    # Only plot price with buy/sell trades
    plot_price_with_trades(
        df,
        portfolio.history,
        params.get("pair_name", "Crypto/USDC"),
        threshold_pct=params.get("threshold_pct", None)
    )
    return portfolio
