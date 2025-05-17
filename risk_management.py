"""
risk_management.py
Functions for setting and monitoring stop-loss and take-profit levels in a trading strategy.
"""
import pandas as pd


def set_stop_loss_take_profit(entry_price, stop_loss_pct, take_profit_pct, side):
    """
    Calculate stop-loss and take-profit prices for a trade.
    If stop_loss_pct or take_profit_pct is None, return None for that price.
    """
    if side == 'buy':
        stop_loss_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct is not None else None
        take_profit_price = entry_price * (1 + take_profit_pct) if take_profit_pct is not None else None
    elif side == 'sell':
        stop_loss_price = entry_price * (1 + stop_loss_pct) if stop_loss_pct is not None else None
        take_profit_price = entry_price * (1 - take_profit_pct) if take_profit_pct is not None else None
    else:
        raise ValueError("side must be 'buy' or 'sell'")
    return stop_loss_price, take_profit_price


def check_stop_loss_take_profit(row, position):
    """
    Check if stop-loss or take-profit is triggered for a position on the current candle.
    Args:
        row (pd.Series): Current OHLCV row.
        position (dict): Open position with keys: 'side', 'stop_loss_price', 'take_profit_price'.
    Returns:
        exit_reason (str or None): 'stop_loss', 'take_profit', or None.
    """
    # Defensive: If stop_loss_price or take_profit_price is None, skip those checks
    if position['side'] == 'buy':
        # Stop-loss: if low <= stop_loss_price
        if position.get('stop_loss_price') is not None and position['stop_loss_price'] is not None:
            if row['low'] <= position['stop_loss_price']:
                return 'stop_loss'
        # Take-profit: if high >= take_profit_price
        if position.get('take_profit_price') is not None and position['take_profit_price'] is not None:
            if row['high'] >= position['take_profit_price']:
                return 'take_profit'
    elif position['side'] == 'sell':
        # Stop-loss: if high >= stop_loss_price
        if position.get('stop_loss_price') is not None and position['stop_loss_price'] is not None:
            if row['high'] >= position['stop_loss_price']:
                return 'stop_loss'
        # Take-profit: if low <= take_profit_price
        if position.get('take_profit_price') is not None and position['take_profit_price'] is not None:
            if row['low'] <= position['take_profit_price']:
                return 'take_profit'
    return None
