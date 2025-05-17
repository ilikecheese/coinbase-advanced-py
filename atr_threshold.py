"""
atr_threshold.py
Module for calculating ATR and dynamic buy/sell thresholds for trading strategies.
"""
import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) for a DataFrame with OHLC data.
    Args:
        df (pd.DataFrame): DataFrame with columns 'high', 'low', 'close'.
        period (int): ATR period (default 14).
    Returns:
        pd.Series: ATR values.
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def get_dynamic_threshold(df, period=14, k=1.5):
    """
    Calculate dynamic threshold based on ATR as a percentage of price, scaled by k.
    Args:
        df (pd.DataFrame): DataFrame with columns 'high', 'low', 'close'.
        period (int): ATR period.
        k (float): Scaling factor.
    Returns:
        pd.Series: Dynamic threshold values (percent).
    """
    atr = calculate_atr(df, period)
    atr_pct = (atr / df['close']) * 100
    dynamic_threshold = k * atr_pct
    return dynamic_threshold

def update_strategy_with_dynamic_threshold(df, strategy_params):
    """
    Add buy_threshold and sell_threshold columns to DataFrame using dynamic ATR-based thresholds.
    Args:
        df (pd.DataFrame): DataFrame with OHLC data.
        strategy_params (dict): Should include 'atr_period' and 'atr_k'.
    Returns:
        pd.DataFrame: DataFrame with new columns.
    """
    period = strategy_params.get('atr_period', 14)
    k = strategy_params.get('atr_k', 1.5)
    dynamic_threshold = get_dynamic_threshold(df, period=period, k=k)
    df['buy_threshold'] = dynamic_threshold
    df['sell_threshold'] = dynamic_threshold
    return df
