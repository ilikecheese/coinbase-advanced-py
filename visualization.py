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
        # Green: price < base_price (trending toward buy)
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
        # Red: price > base_price (trending toward sell)
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

def plot_portfolio_value_over_time(portfolio_history, pair_name="Crypto/USDC", save_path=None, show=True):
    """
    Plot portfolio value (in USD) over time.
    Args:
        portfolio_history: List of (timestamp, value) tuples or DataFrame with 'date' and 'value' columns
        pair_name: String for chart title
        save_path: Optional path to save the plot as an image
        show: Whether to display the plot interactively (default True)
    """
    if isinstance(portfolio_history, list):
        df = pd.DataFrame(portfolio_history, columns=["date", "value"])
    else:
        df = portfolio_history
    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["value"], color="blue", label="Portfolio Value (USD)")
    plt.title(f"{pair_name} Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Portfolio value plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_price_and_portfolio(df, portfolio_history, pair_name="Crypto/USDC", save_path=None, show=True):
    """
    Plot price candles and portfolio value on two y-axes, sharing the same time axis.
    Args:
        df: DataFrame with 'date' and 'close' columns
        portfolio_history: List of (date, value) tuples or DataFrame with 'date' and 'value' columns
        pair_name: String for chart title
        save_path: Optional path to save the plot as an image
        show: Whether to display the plot interactively (default True)
    """
    import matplotlib.dates as mdates
    if isinstance(portfolio_history, list):
        pf_df = pd.DataFrame(portfolio_history, columns=["date", "value"])
    else:
        pf_df = portfolio_history
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df["date"], df["close"], color="black", label="Price (USDC)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USDC)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # Portfolio value on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(pf_df["date"], pf_df["value"], color="blue", label="Portfolio Value (USD)", alpha=0.7)
    ax2.set_ylabel("Portfolio Value (USD)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    # Title and legend
    fig.suptitle(f"{pair_name}: Price and Portfolio Value Over Time")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path)
        print(f"Combined plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_price_with_trades_and_portfolio(df, trade_history, portfolio_history, pair_name="Crypto/USDC", threshold_pct=None):
    """
    Plot price chart with buy/sell markers and annotate portfolio value at each trade.
    Args:
        df: DataFrame with 'date' and 'close' columns
        trade_history: List of (action, price, amount, fee) tuples
        portfolio_history: List of (date, value) tuples
        pair_name: String for chart title
        threshold_pct: Percentage threshold for buy/sell zones (float or None)
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["close"], label="Price", color="black")
    ax = plt.gca()
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
    # Annotate portfolio USD and crypto value at each trade
    pf_dict = dict(portfolio_history)
    for d, p in zip(buy_dates, buy_prices):
        val = pf_dict.get(d, None)
        if val is not None:
            # Find crypto value at this date
            idx = df[df["date"] == d].index[0]
            price = df.iloc[idx]["close"]
            # Estimate crypto holdings: (portfolio USD value - cash) / price, but we don't have cash directly, so just show both
            ax.annotate(f"${val:.2f}\n@{price:.5f}", (d, p), textcoords="offset points", xytext=(0,10), ha='center', color='green', fontsize=8)
    for d, p in zip(sell_dates, sell_prices):
        val = pf_dict.get(d, None)
        if val is not None:
            idx = df[df["date"] == d].index[0]
            price = df.iloc[idx]["close"]
            ax.annotate(f"${val:.2f}\n@{price:.5f}", (d, p), textcoords="offset points", xytext=(0,-15), ha='center', color='red', fontsize=8)
    plt.title(f"{pair_name} Price with Buy/Sell Trades and Portfolio Value at Trades")
    plt.xlabel("Date")
    plt.ylabel("Price (USDC)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()
