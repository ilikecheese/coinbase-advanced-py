import pandas as pd
from backtest import run_backtest_with_risk_management

def test_stop_loss_take_profit():
    # Create a simple DataFrame with a price drop and rise to trigger stop-loss and take-profit
    data = {
        'date': pd.date_range(start='2025-01-01', periods=6, freq='T'),
        'open': [100, 99, 98, 97, 98, 99],
        'high': [100, 100, 99, 98, 99, 100],
        'low':  [100, 98, 97, 96, 97, 98],
        'close':[100, 99, 98, 97, 98, 99],
    }
    df = pd.DataFrame(data)
    params = {
        'initial_cash': 1000,
        'initial_crypto': 0,
        'fee_pct': 0.0,
        'threshold_pct': 1.0,
        'min_cash_reserve': 0,
        'stop_loss_pct': 0.02,   # 2%
        'take_profit_pct': 0.02, # 2%
    }
    trade_log, portfolio_value = run_backtest_with_risk_management(df, params)
    print('Trade log:')
    for trade in trade_log:
        print(trade)
    print(f'Final portfolio value: {portfolio_value}')
    # Check that at least one stop-loss or take-profit exit occurred
    assert any(trade.get('exit_reason') in ('stop_loss', 'take_profit') for trade in trade_log if trade['action'] in ('sell', 'exit'))

def main():
    test_stop_loss_take_profit()
    print('Stop-loss/take-profit test passed.')

if __name__ == '__main__':
    main()
