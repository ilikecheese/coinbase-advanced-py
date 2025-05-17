import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR) for a DataFrame with OHLC data."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def get_dynamic_threshold(df, period=14, k=1.5):
    """Calculate dynamic threshold based on ATR."""
    atr = calculate_atr(df, period)
    # Convert ATR to percentage of price
    atr_pct = (atr / df['close']) * 100
    # Scale by factor k
    dynamic_threshold = k * atr_pct
    return dynamic_threshold

def update_strategy_with_dynamic_threshold(df, strategy_params):
    """Update strategy parameters with dynamic thresholds."""
    dynamic_threshold = get_dynamic_threshold(df, period=strategy_params.get('atr_period', 14),
                                             k=strategy_params.get('atr_k', 1.5))
    df['buy_threshold'] = dynamic_threshold
    df['sell_threshold'] = dynamic_threshold
    return df

# Example usage in your backtest system
def run_backtest_with_dynamic_threshold(data, strategy_params):
    """Modified backtest function to use dynamic thresholds."""
    data = update_strategy_with_dynamic_threshold(data, strategy_params)
    
    # Example: Modify your existing strategy logic
    for i in range(1, len(data)):
        last_price = data['last_trade_price'].iloc[i-1]
        current_price = data['close'].iloc[i]
        buy_threshold = data['buy_threshold'].iloc[i]
        sell_threshold = data['sell_threshold'].iloc[i]
        
        # Buy if price drops by buy_threshold% and cash is available
        if (last_price - current_price) / last_price * 100 >= buy_threshold:
            # Execute buy (your existing logic)
            pass
        # Sell if price rises by sell_threshold% and crypto is available
        elif (current_price - last_price) / last_price * 100 >= sell_threshold:
            # Execute sell (your existing logic)
            pass
    
    return data