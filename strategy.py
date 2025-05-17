import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from risk_management import set_stop_loss_take_profit, check_stop_loss_take_profit

def should_buy(last_trade_price, current_price, threshold_pct):
    return (current_price <= last_trade_price * (1 - threshold_pct / 100))

def should_sell(last_trade_price, current_price, threshold_pct):
    return (current_price >= last_trade_price * (1 + threshold_pct / 100))

def find_optimal_trades(df, prominence=0.0):
    """
    Identify optimal buy/sell points using local minima/maxima.
    Returns a list of (action, price, qty, fee, date) tuples.
    """
    prices = df["close"].values
    # Find local minima (buys) and maxima (sells)
    local_min_idx = argrelextrema(prices, np.less)[0]
    local_max_idx = argrelextrema(prices, np.greater)[0]
    trades = []
    # Assume we start with cash, buy at first min, alternate buy/sell
    holding_crypto = False
    qty = 1  # For comparison, use 1 unit per trade (or could simulate full portfolio)
    fee = 0  # Ignore fees for optimal for now, or pass as param
    last_action = None
    for i in range(len(df)):
        if i in local_min_idx and (last_action != "buy"):
            trades.append(("buy", prices[i], qty, fee, df["date"].iloc[i]))
            last_action = "buy"
        elif i in local_max_idx and (last_action != "sell"):
            trades.append(("sell", prices[i], qty, fee, df["date"].iloc[i]))
            last_action = "sell"
    return trades

def find_optimal_strategy_trades(df, threshold_pct, initial_cash, initial_crypto, min_cash_reserve, fee_pct):
    """
    Simulate optimal trades using the same strategy parameters as the actual backtest,
    but with perfect hindsight (always buying at the lowest and selling at the highest within each threshold swing).
    Returns a trade history list: (action, price, qty, fee, date)
    """
    cash = initial_cash
    crypto = initial_crypto / df.iloc[0]["close"]  # Convert initial_crypto (USD) to crypto units at start
    last_trade_price = df.iloc[0]["close"]
    trades = []
    holding_crypto = True  # Start with both cash and crypto
    for idx, row in df.iterrows():
        price = row["close"]
        date = row["date"]
        # Buy logic (price drops by threshold_pct from last trade)
        if cash > min_cash_reserve and price <= last_trade_price * (1 - threshold_pct / 100):
            pct = threshold_pct / 100
            amount_to_spend = cash * pct
            fee = amount_to_spend * fee_pct / 100
            qty = (amount_to_spend - fee) / price
            if amount_to_spend > 0:
                cash -= amount_to_spend
                crypto += qty
                trades.append(("buy", price, qty, fee, date))
                last_trade_price = price
        # Sell logic (price rises by threshold_pct from last trade)
        elif crypto > 0 and price >= last_trade_price * (1 + threshold_pct / 100):
            pct = threshold_pct / 100
            qty = crypto * pct
            proceeds = qty * price
            fee = proceeds * fee_pct / 100
            if qty > 0:
                cash += proceeds - fee
                crypto -= qty
                trades.append(("sell", price, qty, fee, date))
                last_trade_price = price
    return trades

def run_strategy_with_risk_management(df, strategy_params):
    """
    Run backtest with stop-loss and take-profit logic integrated.
    Returns trade log with entry/exit details and updated portfolio.
    """
    initial_cash = strategy_params["initial_cash"]
    initial_crypto = strategy_params["initial_crypto"]
    fee_pct = strategy_params["fee_pct"]
    stop_loss_pct = strategy_params["stop_loss_pct"]
    take_profit_pct = strategy_params["take_profit_pct"]
    min_cash_reserve = strategy_params["min_cash_reserve"]
    threshold_pct = strategy_params["threshold_pct"]

    # Convert string 'None' to Python None for robustness
    if stop_loss_pct == 'None':
        stop_loss_pct = None
    if take_profit_pct == 'None':
        take_profit_pct = None

    cash = initial_cash
    crypto = initial_crypto / df.iloc[0]["close"] if initial_crypto > 0 else 0
    last_trade_price = df.iloc[0]["close"]
    open_position = None
    trade_log = []

    for idx, row in df.iterrows():
        price = row["close"]
        date = row["date"]
        # Check for stop-loss/take-profit exit first
        if open_position:
            exit_reason = check_stop_loss_take_profit(row, open_position)
            if exit_reason:
                # Close position
                side = open_position['side']
                qty = open_position['qty']
                entry_price = open_position['entry_price']
                stop_loss_price = open_position['stop_loss_price']
                take_profit_price = open_position['take_profit_price']
                if side == 'buy':
                    proceeds = qty * price
                    fee = proceeds * fee_pct / 100
                    cash += proceeds - fee
                    crypto -= qty
                else:  # 'sell' (for shorting, if supported)
                    cost = qty * price
                    fee = cost * fee_pct / 100
                    cash -= cost + fee
                    crypto += qty
                trade_log.append({
                    'action': 'exit',
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'qty': qty,
                    'fee': fee,
                    'date': date,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'exit_reason': exit_reason
                })
                open_position = None
                last_trade_price = price
                continue
        # Entry logic (buy)
        if not open_position and cash > min_cash_reserve and price <= last_trade_price * (1 - threshold_pct / 100):
            pct = threshold_pct / 100
            amount_to_spend = cash * pct
            fee = amount_to_spend * fee_pct / 100
            qty = (amount_to_spend - fee) / price
            stop_loss_price, take_profit_price = set_stop_loss_take_profit(price, stop_loss_pct, take_profit_pct, 'buy')
            cash -= amount_to_spend
            crypto += qty
            open_position = {
                'side': 'buy',
                'entry_price': price,
                'qty': qty,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price
            }
            trade_log.append({
                'action': 'buy',
                'entry_price': price,
                'qty': qty,
                'fee': fee,
                'date': date,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price
            })
            last_trade_price = price
        # Exit logic (sell)
        elif open_position and open_position['side'] == 'buy' and crypto > 0 and price >= last_trade_price * (1 + threshold_pct / 100):
            qty = open_position['qty']
            proceeds = qty * price
            fee = proceeds * fee_pct / 100
            cash += proceeds - fee
            crypto -= qty
            trade_log.append({
                'action': 'sell',
                'entry_price': open_position['entry_price'],
                'exit_price': price,
                'qty': qty,
                'fee': fee,
                'date': date,
                'stop_loss_price': open_position['stop_loss_price'],
                'take_profit_price': open_position['take_profit_price'],
                'exit_reason': 'take_profit' if open_position['take_profit_price'] is not None and price >= open_position['take_profit_price'] else 'manual_sell'
            })
            open_position = None
            last_trade_price = price
    return trade_log
