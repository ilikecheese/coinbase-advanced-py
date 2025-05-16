def should_buy(last_trade_price, current_price, threshold_pct):
    return (current_price <= last_trade_price * (1 - threshold_pct / 100))

def should_sell(last_trade_price, current_price, threshold_pct):
    return (current_price >= last_trade_price * (1 + threshold_pct / 100))
