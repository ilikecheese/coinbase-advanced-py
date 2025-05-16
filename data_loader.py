import pandas as pd

def load_candle_data(csv_file):
    """Load 1-minute candle data from CSV. Expects columns: date, open, high, low, close, volume."""
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df = df.sort_values("date")
    return df
