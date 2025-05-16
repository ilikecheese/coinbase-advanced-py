class Portfolio:
    def __init__(self, initial_cash, initial_crypto, fee_pct=0.0):
        self.cash = initial_cash
        self.crypto = initial_crypto
        self.fee_pct = fee_pct
        self.history = []

    def buy(self, price, pct):
        amount_to_spend = self.cash * pct
        fee = amount_to_spend * self.fee_pct / 100
        crypto_bought = (amount_to_spend - fee) / price
        self.cash -= amount_to_spend
        self.crypto += crypto_bought
        self.history.append(("buy", price, crypto_bought, fee))

    def sell(self, price, pct):
        crypto_to_sell = self.crypto * pct
        proceeds = crypto_to_sell * price
        fee = proceeds * self.fee_pct / 100
        self.cash += proceeds - fee
        self.crypto -= crypto_to_sell
        self.history.append(("sell", price, crypto_to_sell, fee))

    def value(self, price):
        return self.cash + self.crypto * price

    def value_history(self, df):
        """
        Returns a list of (date, value) tuples for the portfolio value over time.
        Assumes self.history is in order and df is a DataFrame with 'date' and 'close'.
        """
        values = []
        cash = self.cash
        crypto = self.crypto
        fee_pct = self.fee_pct
        # Rewind to initial state
        cash = self.history[0][2] * self.history[0][1] if self.history and self.history[0][0] == 'buy' else cash
        crypto = self.history[0][2] if self.history and self.history[0][0] == 'buy' else crypto
        cash = self.history[0][2] * self.history[0][1] if self.history and self.history[0][0] == 'sell' else cash
        crypto = self.history[0][2] if self.history and self.history[0][0] == 'sell' else crypto
        # Replay trades
        cash = self.cash
        crypto = self.crypto
        idx = 0
        for i, row in df.iterrows():
            price = row['close']
            date = row['date']
            # Apply trades up to this point
            while idx < len(self.history) and abs(price - self.history[idx][1]) < 1e-10:
                action, trade_price, amount, fee = self.history[idx]
                if action == 'buy':
                    cash -= amount * trade_price + fee
                    crypto += amount
                elif action == 'sell':
                    cash += amount * trade_price - fee
                    crypto -= amount
                idx += 1
            values.append((date, cash + crypto * price))
        return values
