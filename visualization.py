import matplotlib.pyplot as plt
import pandas as pd

def plot_price_with_trades(df, trade_history, pair_name="Crypto/USDC", threshold_pct=None):
    """
    Plot price chart with buy/sell markers from trade history. (Zone shading temporarily disabled.)
    Args:
        df: DataFrame with 'date' and 'close' columns
        trade_history: List of (action, price, amount, fee) tuples
        pair_name: String for chart title
        threshold_pct: Percentage threshold for buy/sell zones (float or None)
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["close"], label="Price", color="black")
    ax = plt.gca()
    # --- Zone shading temporarily disabled ---
    # if threshold_pct is not None:
    #     base_price = df["close"].iloc[0]
    #     in_green = False
    #     in_red = False
    #     start_idx = None
    #     # Green: price < base_price (trending toward buy)
    #     for i, price in enumerate(df["close"]):
    #         if price < base_price:
    #             if not in_green:
    #                 in_green = True
    #                 start_idx = i
    #         else:
    #             if in_green:
    #                 ax.axvspan(df["date"].iloc[start_idx], df["date"].iloc[i-1], color="green", alpha=0.08, label="Trending to Buy" if start_idx == 0 else None)
    #                 in_green = False
    #     if in_green:
    #         ax.axvspan(df["date"].iloc[start_idx], df["date"].iloc[-1], color="green", alpha=0.08, label="Trending to Buy" if start_idx == 0 else None)
    #     # Red: price > base_price (trending toward sell)
    #     for i, price in enumerate(df["close"]):
    #         if price > base_price:
    #             if not in_red:
    #                 in_red = True
    #                 start_idx = i
    #         else:
    #             if in_red:
    #                 ax.axvspan(df["date"].iloc[start_idx], df["date"].iloc[i-1], color="red", alpha=0.08, label="Trending to Sell" if start_idx == 0 else None)
    #                 in_red = False
    #     if in_red:
    #         ax.axvspan(df["date"].iloc[start_idx], df["date"].iloc[-1], color="red", alpha=0.08, label="Trending to Sell" if start_idx == 0 else None)
    # Extract buy/sell points
    buy_points = [(t[1], t[2]) for t in trade_history if t[0] == "buy"]
    sell_points = [(t[1], t[2]) for t in trade_history if t[0] == "sell"]
    # Find corresponding dates for trades
    buy_dates = [df[df["close"] == price]["date"].iloc[0] for price, _ in buy_points if not df[df["close"] == price].empty]
    buy_prices = [price for price, _ in buy_points if not df[df["close"] == price].empty]
    sell_dates = [df[df["close"] == price]["date"].iloc[0] for price, _ in sell_points if not df[df["close"] == price].empty]
    sell_prices = [price for price, _ in sell_points if not df[df["close"] == price].empty]
    plt.scatter(buy_dates, buy_prices, marker="^", color="green", label="Buy", s=80, zorder=5)
    plt.scatter(sell_dates, sell_prices, marker="v", color="red", label="Sell", s=80, zorder=5)
    plt.title(f"{pair_name} Price with Buy/Sell Trades")
    plt.xlabel("Date")
    plt.ylabel("Price (USDC)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()
