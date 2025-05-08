# Crypto Sideways Market Analyzer

## Overview

This tool analyzes cryptocurrencies on Coinbase to identify those with high volatility within sideways (range-bound) markets. It finds assets that have significant price movement within established support and resistance levels, which can be ideal candidates for range trading strategies.

## Features

- **Automated Data Collection**: Fetches historical price data from Coinbase for all USD cryptocurrency pairs
- **Smart Granularity Selection**: Automatically calculates optimal data point spacing to maximize data collection within API limits
- **Volatility Analysis**: Calculates annualized volatility based on price returns
- **Sideways Market Detection**: Identifies non-trending markets using price range variance and moving average slope
- **Support & Resistance Identification**: Uses peak detection and clustering algorithms to find consistent price levels
- **Touch Point Counting**: Counts how many times price has interacted with support/resistance levels
- **Test Mode**: Limit analysis to a specific number of currencies for faster testing

## Installation

The script uses the Coinbase Advanced Trade Python SDK along with data analysis libraries:

```bash
pip install pandas numpy scipy matplotlib
```

Make sure you have a `config.py` file with your Coinbase API credentials:

```python
API_KEY = "your_coinbase_api_key"
API_SECRET = "your_coinbase_api_secret"
```

## Usage

Basic usage with default settings:

```bash
python crypto_sideways_analyzer.py
```

### Command-line Parameters

- `--start`: Start date for analysis (YYYY-MM-DD), default: 1 year ago
- `--end`: End date for analysis (YYYY-MM-DD), default: today
- `--volatility`: Minimum volatility threshold as decimal (default: 0.02 or 2%)
- `--tolerance`: Price touch tolerance for support/resistance (default: 0.01 or 1%)
- `--min-touches`: Minimum number of touches for valid support/resistance (default: 3)
- `--limit`: Maximum number of results to display (default: 5)
- `--test-limit`: Limit number of cryptocurrencies to analyze (default: 10, 0 means analyze all)
- `--granularity`: Candle time interval (default: auto-calculated based on date range)

### Granularity Options

- ONE_MINUTE: 1-minute candles
- FIVE_MINUTE: 5-minute candles
- FIFTEEN_MINUTE: 15-minute candles
- THIRTY_MINUTE: 30-minute candles
- ONE_HOUR: 1-hour candles
- TWO_HOUR: 2-hour candles
- SIX_HOUR: 6-hour candles
- ONE_DAY: 1-day candles

By default, the script automatically selects the optimal granularity based on your date range to maximize data points while staying within Coinbase's 350 candle limit.

## Examples

### Quick test with 5 cryptocurrencies

```bash
python crypto_sideways_analyzer.py --test-limit 5
```

### Analyze a specific time period with auto-granularity

```bash
python crypto_sideways_analyzer.py --start 2025-01-01 --end 2025-05-01
```

### Custom analysis parameters

```bash
python crypto_sideways_analyzer.py --volatility 0.03 --tolerance 0.015 --min-touches 4
```

### Force specific granularity

```bash
python crypto_sideways_analyzer.py --granularity ONE_HOUR
```

### Full analysis of all cryptocurrencies

```bash
python crypto_sideways_analyzer.py --test-limit 0
```

## Sample Output

```
====================================================================================================
CRYPTO SIDEWAYS MARKET ANALYSIS (2025-01-01 to 2025-05-01)
====================================================================================================
PRODUCT    VOLATILITY   SUPPORT                RESISTANCE              RANGE %   
----------------------------------------------------------------------------------------------------
ETH-USD    2.80%        $3000.00 (10 touches)  $3500.00 (8 touches)   16.67%
SOL-USD    2.65%        $85.50 (6 touches)     $112.30 (5 touches)    31.34%
DOT-USD    2.45%        $6.30 (7 touches)      $8.10 (4 touches)      28.57%
----------------------------------------------------------------------------------------------------

BEST CANDIDATE: ETH-USD
Volatility: 2.80%
Support level: $3000.00 (touched 10 times)
Resistance level: $3500.00 (touched 8 times)
Current price: $3250.75
Time frame analyzed: 2025-01-01 to 2025-05-01
====================================================================================================
```

## Understanding the Results

- **Volatility**: Higher values indicate more price movement (calculated as annualized standard deviation of returns)
- **Support/Resistance**: Price levels where buying/selling pressure consistently appears
- **Touches**: Number of times price has interacted with these levels (higher = more reliable)
- **Range %**: Size of the trading channel as a percentage (higher = more trading opportunity between levels)

## API Rate Limiting

The script includes built-in delays between API calls to respect Coinbase's rate limits. If you encounter rate limiting issues, try increasing the delay in the fetch_historical_data method.