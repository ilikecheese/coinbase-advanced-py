import matplotlib.pyplot as plt
import pandas as pd

def plot_price_with_trades(df, trade_history, pair_name="Crypto/USDC", threshold_pct=None, save_path=None, optimal_trades=None):
    """
    Plot price chart with buy/sell markers from trade history. Optionally overlay optimal trades.
    Args:
        df: DataFrame with 'date' and 'close' columns
        trade_history: List of (action, price, amount, fee) tuples
        pair_name: String for chart title
        threshold_pct: Percentage threshold for buy/sell zones (float or None)
        save_path: Optional path to save the plot as an image
        optimal_trades: List of (action, price, qty, fee, date) tuples for optimal trades
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["close"], label="Price", color="black")
    ax = plt.gca()
    # Probable trades (actual)
    buy_points = [(t[1], t[2]) for t in trade_history if t[0] == "buy"]
    sell_points = [(t[1], t[2]) for t in trade_history if t[0] == "sell"]
    buy_dates = [df[df["close"] == price]["date"].iloc[0] for price, _ in buy_points if not df[df["close"] == price].empty]
    buy_prices = [price for price, _ in buy_points if not df[df["close"] == price].empty]
    sell_dates = [df[df["close"] == price]["date"].iloc[0] for price, _ in sell_points if not df[df["close"] == price].empty]
    sell_prices = [price for price, _ in sell_points if not df[df["close"] == price].empty]
    plt.scatter(buy_dates, buy_prices, marker="^", color="green", label="Buy (Probable)", s=80, zorder=5)
    plt.scatter(sell_dates, sell_prices, marker="v", color="red", label="Sell (Probable)", s=80, zorder=5)
    # Optimal trades overlay
    if optimal_trades is not None:
        opt_buy_dates = [t[4] for t in optimal_trades if t[0] == "buy"]
        opt_buy_prices = [t[1] for t in optimal_trades if t[0] == "buy"]
        opt_sell_dates = [t[4] for t in optimal_trades if t[0] == "sell"]
        opt_sell_prices = [t[1] for t in optimal_trades if t[0] == "sell"]
        plt.scatter(opt_buy_dates, opt_buy_prices, marker="o", color="lime", label="Buy (Optimal)", s=60, edgecolor='black', zorder=6)
        plt.scatter(opt_sell_dates, opt_sell_prices, marker="x", color="orange", label="Sell (Optimal)", s=80, zorder=6)
    plt.title(f"{pair_name} Price with Buy/Sell Trades (Probable vs Optimal)")
    plt.xlabel("Date")
    plt.ylabel("Price (USDC)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def plot_price_with_trade_log(df, trade_log, pair_name="Crypto/USDC", save_path=None):
    """
    Plot price chart with buy/sell/stop-loss/take-profit markers from trade log.
    Args:
        df: DataFrame with 'date' and 'close' columns
        trade_log: List of trade dicts with 'action', 'entry_price', 'exit_price', 'date', 'exit_reason', etc.
        pair_name: String for chart title
        save_path: Optional path to save the plot as an image
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["close"], label="Price", color="black")
    ax = plt.gca()
    # Buy entries
    buy_dates = [t['date'] for t in trade_log if t['action'] == 'buy']
    buy_prices = [t['entry_price'] for t in trade_log if t['action'] == 'buy']
    plt.scatter(buy_dates, buy_prices, marker="^", color="green", label="Buy", s=80, zorder=5)
    # Sell/exit exits
    sell_dates = [t['date'] for t in trade_log if t['action'] in ('sell', 'exit') and t.get('exit_reason') == 'manual_sell']
    sell_prices = [t['exit_price'] for t in trade_log if t['action'] in ('sell', 'exit') and t.get('exit_reason') == 'manual_sell']
    plt.scatter(sell_dates, sell_prices, marker="v", color="blue", label="Sell", s=80, zorder=5)
    # Stop-loss exits
    sl_dates = [t['date'] for t in trade_log if t['action'] in ('sell', 'exit') and t.get('exit_reason') == 'stop_loss']
    sl_prices = [t['exit_price'] for t in trade_log if t['action'] in ('sell', 'exit') and t.get('exit_reason') == 'stop_loss']
    plt.scatter(sl_dates, sl_prices, marker="x", color="red", label="Stop-Loss Exit", s=100, zorder=6)
    # Take-profit exits
    tp_dates = [t['date'] for t in trade_log if t['action'] in ('sell', 'exit') and t.get('exit_reason') == 'take_profit']
    tp_prices = [t['exit_price'] for t in trade_log if t['action'] in ('sell', 'exit') and t.get('exit_reason') == 'take_profit']
    plt.scatter(tp_dates, tp_prices, marker="*", color="gold", label="Take-Profit Exit", s=120, zorder=6)
    plt.title(f"{pair_name} Price with Trade Exits (Stop-Loss/Take-Profit)")
    plt.xlabel("Date")
    plt.ylabel("Price (USDC)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()
