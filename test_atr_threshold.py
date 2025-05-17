import pandas as pd
from atr_threshold import calculate_atr, get_dynamic_threshold, update_strategy_with_dynamic_threshold

def test_atr_threshold():
    # Create a simple DataFrame with synthetic OHLC data
    data = {
        'high': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        'low':  [ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'close':[ 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5],
    }
    df = pd.DataFrame(data)
    # Test ATR calculation
    atr = calculate_atr(df, period=3)
    print('ATR:')
    print(atr)
    # Test dynamic threshold calculation
    dyn_thresh = get_dynamic_threshold(df, period=3, k=1.5)
    print('Dynamic Threshold:')
    print(dyn_thresh)
    # Test DataFrame update
    strategy_params = {'atr_period': 3, 'atr_k': 1.5}
    df2 = update_strategy_with_dynamic_threshold(df.copy(), strategy_params)
    print('DataFrame with thresholds:')
    print(df2[['close', 'buy_threshold', 'sell_threshold']])
    # Check that columns exist and are not all NaN
    assert 'buy_threshold' in df2.columns and 'sell_threshold' in df2.columns
    assert df2['buy_threshold'].notna().sum() > 0
    assert df2['sell_threshold'].notna().sum() > 0
    print('ATR threshold module test passed.')

if __name__ == '__main__':
    test_atr_threshold()
