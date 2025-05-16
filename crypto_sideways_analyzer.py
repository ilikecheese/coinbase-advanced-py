#!/usr/bin/env python3
"""
Crypto Sideways Market Analyzer

This script identifies volatile cryptocurrencies in sideways (range-bound) markets on Coinbase.
It analyzes historical price data to find cryptocurrencies with high volatility
but within stable price channels, and counts how many times price touches support and resistance levels.

Features:
- Fetches historical OHLC data from Coinbase API
- Calculates volatility using standard deviation of daily returns
- Detects sideways markets using rolling statistics
- Identifies support and resistance levels
- Counts touches at these levels
- Automatically selects optimal granularity based on date range
- Visualizes price action with support/resistance levels and touch points
- Saves historical data and analysis results to spreadsheets
"""

import os
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import signal
from scipy.cluster.vq import kmeans
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch
import mplfinance as mpf
import json
import csv
from pathlib import Path
import openpyxl

# ===============================================================================
# DEFAULT CONFIGURATION - MODIFY THESE VALUES TO CHANGE DEFAULT BEHAVIOR
# ===============================================================================

# Date range defaults
DEFAULT_DAYS_LOOKBACK = 5  # Changed to 5 days for longer analysis period
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')  # Today
DEFAULT_START_DATE = (datetime.now() - timedelta(days=DEFAULT_DAYS_LOOKBACK)).strftime('%Y-%m-%d')

# Analysis parameters defaults
DEFAULT_VOLATILITY_THRESHOLD = 0.02  # 2% minimum volatility
DEFAULT_SIDEWAYS_TOLERANCE = 0.1  # 10% maximum range variance to qualify as sideways
DEFAULT_TOUCH_TOLERANCE = 0.01  # 1% tolerance around support/resistance for "touches"
DEFAULT_MIN_TOUCHES = 3  # Minimum number of touches to consider valid support/resistance
DEFAULT_RESULTS_LIMIT = 5  # Number of top results to display

# Cryptocurrency data collection defaults
DEFAULT_TEST_LIMIT = 10  # Limit analysis to this many cryptocurrencies (0 for all)
DEFAULT_HISTORY_PERIODS = 5  # Number of chunks to fetch when collecting history
DEFAULT_CHUNK_SIZE_DAYS = .1  # Set to 1 day per chunk for 5-minute candles (288 candles per chunk is within API limit)

# Granularity choices (time interval between candles)
DEFAULT_GRANULARITY = "FIVE_MINUTE"  # Set to FIVE_MINUTE to always use 5-minute candles

# Available granularity options and their API values:
# "ONE_MINUTE"     - 1 minute candles
# "FIVE_MINUTE"    - 5 minute candles
# "FIFTEEN_MINUTE" - 15 minute candles
# "THIRTY_MINUTE"  - 30 minute candles
# "ONE_HOUR"       - 1 hour candles
# "TWO_HOUR"       - 2 hour candles
# "SIX_HOUR"       - 6 hour candles
# "ONE_DAY"        - 1 day candles

# Maximum candles per granularity to stay within API limits:
# ONE_MINUTE:      5.8 hours (350 minutes)
# FIVE_MINUTE:     29 hours (1 day + 5 hours)
# FIFTEEN_MINUTE:  3.6 days
# THIRTY_MINUTE:   7.3 days
# ONE_HOUR:        14.6 days
# TWO_HOUR:        29.2 days
# SIX_HOUR:        87.5 days
# ONE_DAY:         350 days

# Output format for saving data
DEFAULT_FILE_FORMAT = 'csv'  # Currently only 'csv' is supported

# Directory names
DATA_DIR = "crypto_data"
PLOTS_DIR = "plots"
COMBINED_MASTER_FILENAME = "all_currencies_master.csv"

# ===============================================================================
# Configure logging
# ===============================================================================
import logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG for more detailed logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crypto_sideways_analyzer')

class CryptoSidewaysAnalyzer:
    """Analyzes cryptocurrencies for sideways market patterns with volatility, using only local CSV data."""
    
    # Define granularity options with their duration in minutes
    GRANULARITY_OPTIONS = {
        "ONE_MINUTE": 1,
        "FIVE_MINUTE": 5,
        "FIFTEEN_MINUTE": 15,
        "THIRTY_MINUTE": 30,
        "ONE_HOUR": 60,
        "TWO_HOUR": 120,
        "SIX_HOUR": 360,
        "ONE_DAY": 1440  # 24 hours * 60 minutes
    }
    
    # Maximum candles allowed by Coinbase API
    MAX_CANDLES = 350
    
    def __init__(self, data_dir="market_data", symbols=None, start_date=None, end_date=None, 
                volatility_threshold=0.02, sideways_tolerance=0.1, 
                touch_tolerance=0.01, min_touches=3, test_limit=0):
        """
        Initialize the analyzer with parameters and local data directory
        
        Args:
            data_dir: Directory containing market data (default: market_data)
            symbols: List of symbols to analyze (e.g., ['BTC-USDC']) or None for all
            start_date: Start date for analysis (YYYY-MM-DD) or None for all
            end_date: End date for analysis (YYYY-MM-DD) or None for all
            volatility_threshold: Minimum volatility (std dev of returns) to consider
            sideways_tolerance: Maximum variance in price range to qualify as sideways
            touch_tolerance: Percentage tolerance for price "touches" on support/resistance
            min_touches: Minimum number of touches to consider valid support/resistance
            test_limit: Limit the number of cryptocurrencies to analyze for testing
        """
        self.data_dir = data_dir
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.volatility_threshold = volatility_threshold
        self.sideways_tolerance = sideways_tolerance
        self.touch_tolerance = touch_tolerance
        self.min_touches = min_touches
        self.test_limit = test_limit
        self.results = []

    def get_available_symbols(self):
        """Return all symbols with master_data.csv in the data_dir."""
        symbols = []
        for entry in os.listdir(self.data_dir):
            symbol_dir = os.path.join(self.data_dir, entry)
            if os.path.isdir(symbol_dir):
                master_path = os.path.join(symbol_dir, "master_data.csv")
                if os.path.exists(master_path):
                    symbols.append(entry.replace('_', '-'))
        return symbols

    def load_local_data(self, symbol):
        """Load master_data.csv for a symbol, filter by date if needed."""
        symbol_dir = os.path.join(self.data_dir, symbol.replace('-', '_'))
        master_path = os.path.join(symbol_dir, "master_data.csv")
        if not os.path.exists(master_path):
            logger.warning(f"No local data found for {symbol}")
            return None
        df = pd.read_csv(master_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            if self.start_date:
                df = df[df['date'] >= pd.to_datetime(self.start_date)]
            if self.end_date:
                df = df[df['date'] <= pd.to_datetime(self.end_date)]
            df = df.set_index('date')
        return df

    def calculate_volatility(self, df):
        """Calculate volatility as the standard deviation of returns"""
        if df is None or len(df) < 10:  # Need enough data points
            return 0
            
        # Standard deviation of returns, annualized for daily data
        volatility = df['return'].std() * np.sqrt(252)  # Annualized
        return volatility
        
    def is_sideways_market(self, df):
        """
        Determine if a market is in a sideways pattern
        
        Returns:
            tuple: (is_sideways, details_dict)
        """
        if df is None or len(df) < 20:  # Need enough data for a meaningful analysis
            return False, {}
            
        # Calculate 20-day rolling high and low
        df['rolling_high'] = df['high'].rolling(window=20).max()
        df['rolling_low'] = df['low'].rolling(window=20).min()
        df['price_range'] = df['rolling_high'] - df['rolling_low']
        df['range_percent'] = df['price_range'] / df['rolling_low']
        
        # Calculate 50-day moving average
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        # Drop NaN values after calculating rolling statistics
        df_clean = df.dropna()
        
        if len(df_clean) < 10:  # Need enough data after dropping NaNs
            return False, {}
            
        # Check if range is stable (variance of range < sideways_tolerance)
        range_variance = df_clean['range_percent'].var()
        
        # Check if trend is flat (slope of MA near zero)
        # Calculate the slope of the 50-day moving average
        if len(df_clean) >= 20:
            ma_values = df_clean['ma50'].values[-20:]
            slope = np.polyfit(range(len(ma_values)), ma_values, 1)[0]
            # Normalize slope by average price
            norm_slope = slope / df_clean['ma50'].mean()
        else:
            norm_slope = 1.0  # Default to non-flat if not enough data
            
        # Consider it sideways if range variance is low and slope is near zero
        is_sideways = (range_variance < self.sideways_tolerance and abs(norm_slope) < 0.001)
        
        details = {
            'range_variance': range_variance,
            'normalized_slope': norm_slope,
            'average_range_percent': df_clean['range_percent'].mean(),
            'is_sideways': is_sideways
        }
        
        return is_sideways, details
        
    def identify_support_resistance(self, df):
        """
        Identify support and resistance levels using peak detection and clustering
        
        Returns:
            tuple: (support_levels, resistance_levels)
        """
        if df is None or len(df) < 30:
            return [], []
            
        # Find peaks (local maxima) for resistance levels
        price_series = df['high'].values
        # Use scipy.signal.find_peaks with appropriate distance and prominence
        high_peaks, _ = signal.find_peaks(price_series, distance=5, prominence=price_series.mean() * 0.01)
        
        # Find troughs (local minima) for support levels
        # For troughs, we invert the low price series
        low_series = -df['low'].values
        low_peaks, _ = signal.find_peaks(low_series, distance=5, prominence=(-low_series).mean() * 0.01)
        
        # Extract peak and trough prices
        resistance_prices = [df['high'].values[i] for i in high_peaks]
        support_prices = [df['low'].values[i] for i in low_peaks]
        
        # Cluster similar levels to find consistent zones
        resistance_levels = []
        if resistance_prices:
            try:
                # Use k-means clustering to group similar resistance levels
                # The number of clusters is estimated by taking unique prices with some tolerance
                unique_resistance = len(set([round(p, 1) for p in resistance_prices]))
                k = min(max(2, unique_resistance // 2), 5)  # Limit between 2-5 clusters
                centroids, _ = kmeans(np.array(resistance_prices).reshape(-1, 1), k)
                resistance_levels = [float(c[0]) for c in centroids]
                resistance_levels.sort(reverse=True)  # High to low
            except Exception as e:
                logger.warning(f"Error clustering resistance levels: {e}")
        
        support_levels = []
        if support_prices:
            try:
                # Similar approach for support levels
                unique_support = len(set([round(p, 1) for p in support_prices]))
                k = min(max(2, unique_support // 2), 5)  # Limit between 2-5 clusters
                centroids, _ = kmeans(np.array(support_prices).reshape(-1, 1), k)
                support_levels = [float(c[0]) for c in centroids]
                support_levels.sort()  # Low to high
            except Exception as e:
                logger.warning(f"Error clustering support levels: {e}")
                
        return support_levels, resistance_levels
        
    def count_touches(self, df, level, is_support=True):
        """
        Count how many times price comes within touch_tolerance of a support/resistance level
        
        Args:
            df: DataFrame with price data
            level: The price level to check
            is_support: True for support level, False for resistance
            
        Returns:
            int: Number of touches
        """
        if df is None:
            return 0
            
        tolerance = level * self.touch_tolerance
        
        if is_support:
            # For support, check if low price is within tolerance of the level
            touches = sum((df['low'] >= level - tolerance) & (df['low'] <= level + tolerance))
        else:
            # For resistance, check if high price is within tolerance of the level
            touches = sum((df['high'] >= level - tolerance) & (df['high'] <= level + tolerance))
            
        return touches
        
    def analyze_product(self, symbol):
        """Analyze a single symbol using local data only."""
        df = self.load_local_data(symbol)
        if df is None or len(df) < 30:
            return None
            
        # Calculate volatility
        volatility = self.calculate_volatility(df)
        
        # Check if volatility meets threshold
        if volatility < self.volatility_threshold:
            logger.info(f"{symbol} volatility {volatility:.2%} below threshold {self.volatility_threshold:.2%}")
            return None
            
        # Check if market is sideways
        is_sideways, sideways_details = self.is_sideways_market(df)
        
        if not is_sideways:
            logger.info(f"{symbol} is not in a sideways market")
            return None
            
        # Find support and resistance levels
        support_levels, resistance_levels = self.identify_support_resistance(df)
        
        # Count touches for each level
        support_touches = [(level, self.count_touches(df, level, is_support=True)) 
                         for level in support_levels]
        resistance_touches = [(level, self.count_touches(df, level, is_support=False)) 
                           for level in resistance_levels]
        
        # Filter for levels with minimum touches
        valid_support = [(level, touches) for level, touches in support_touches if touches >= self.min_touches]
        valid_resistance = [(level, touches) for level, touches in resistance_touches if touches >= self.min_touches]
        
        # If we don't have valid levels, this isn't a good candidate
        if not valid_support or not valid_resistance:
            logger.info(f"{symbol} doesn't have strong support/resistance levels")
            return None
            
        # Get the best support and resistance levels (most touches)
        best_support = max(valid_support, key=lambda x: x[1]) if valid_support else (0, 0)
        best_resistance = max(valid_resistance, key=lambda x: x[1]) if valid_resistance else (0, 0)
        
        # Calculate range as percentage
        range_percent = (best_resistance[0] - best_support[0]) / best_support[0] if best_support[0] > 0 else 0
        
        # Prepare results
        result = {
            'product_id': symbol,
            'volatility': volatility,
            'is_sideways': is_sideways,
            'support_level': best_support[0],
            'support_touches': best_support[1],
            'resistance_level': best_resistance[0],
            'resistance_touches': best_resistance[1],
            'price_range_percent': range_percent,
            'average_price': df['close'].mean(),
            'last_price': df['close'].iloc[-1],
            'sideways_details': sideways_details,
            'data_points': len(df)
        }
        
        logger.info(f"Found candidate: {symbol} - Volatility: {volatility:.2%}, "
                   f"Support: ${best_support[0]:.2f} ({best_support[1]} touches), "
                   f"Resistance: ${best_resistance[0]:.2f} ({best_resistance[1]} touches)")
                   
        return result

    def visualize_product(self, product_id, df, support_levels, resistance_levels):
        """
        Visualize the price action with candlestick chart, support/resistance levels, and touch points
        
        Args:
            product_id: The product identifier (e.g., 'BTC-USD')
            df: DataFrame with OHLC data
            support_levels: List of support levels
            resistance_levels: List of resistance levels
        """
        if df is None or len(df) < 10:
            logger.warning(f"Not enough data to visualize {product_id}")
            return
            
        try:
            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Prepare data for visualization - mplfinance requires specific format
            df_plot = df.copy()
            
            # Format the data correctly for mplfinance
            ohlc_data = {
                'Open': df_plot['open'],
                'High': df_plot['high'],
                'Low': df_plot['low'],
                'Close': df_plot['close'],
                'Volume': df_plot['volume']
            }
            
            mpf_df = pd.DataFrame(ohlc_data, index=df_plot.index)
            
            # Add support and resistance levels to the plot
            addplot = []
            for level in support_levels:
                addplot.append(mpf.make_addplot([level] * len(mpf_df), color='blue', linestyle='--', 
                              width=1, label=f'Support: ${level:.2f}'))
            for level in resistance_levels:
                addplot.append(mpf.make_addplot([level] * len(mpf_df), color='red', linestyle='--',
                              width=1, label=f'Resistance: ${level:.2f}'))
            
            # Create the chart title with volatility info
            title = f"{product_id} Price Action - Volatility: {self.calculate_volatility(df):.2%}"
            
            # Plot candlestick chart with mplfinance
            filename = f"plots/{product_id.replace('-', '_')}_{self.start_date}_to_{self.end_date}.png"
            
            # Use mplfinance to create the plot
            mpf.plot(mpf_df, 
                    type='candle', 
                    style='charles',
                    addplot=addplot if addplot else None,
                    title=title,
                    volume=True,
                    figsize=(12, 8),
                    savefig=filename)
            
            logger.info(f"Chart saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error visualizing {product_id}: {e}")
            import traceback
            traceback.print_exc()

    def save_data_to_spreadsheet(self, product_id, df):
        """
        Save the historical price data for a cryptocurrency to CSV file
        
        Args:
            product_id: The product identifier (e.g., 'BTC-USD')
            df: DataFrame with OHLC data
        """
        if df is None or len(df) < 1:
            logger.warning(f"No data to save for {product_id}")
            return
            
        try:
            # Create data directory structure
            base_dir = "crypto_data"
            product_dir = f"{base_dir}/{product_id.replace('-', '_')}"
            Path(product_dir).mkdir(parents=True, exist_ok=True)
            
            # Format the filename with granularity and date range
            granularity_str = self.granularity.lower()
            filename_base = f"{product_id.replace('-', '_')}_{self.start_date}_to_{self.end_date}_{granularity_str}"
            
            # Reset index to make date a column
            df_save = df.copy().reset_index()
            
            # Save to CSV
            csv_path = f"{product_dir}/{filename_base}.csv"
            df_save.to_csv(csv_path, index=False)
            logger.info(f"Data saved to {csv_path}")
            
            # Update the master data file by appending or creating new
            self._update_master_data(product_id, df)
            
        except Exception as e:
            logger.error(f"Error saving data for {product_id}: {e}")
            import traceback
            traceback.print_exc()
            
    def _update_master_data(self, product_id, df):
        """
        Update the master data file for a cryptocurrency by appending new data or creating new file
        
        Args:
            product_id: The product identifier (e.g., 'BTC-USD')
            df: DataFrame with new OHLC data to append
        """
        if df is None or len(df) < 1:
            return
            
        try:
            # Create master data directory structure
            base_dir = "crypto_data"
            product_dir = f"{base_dir}/{product_id.replace('-', '_')}"
            Path(product_dir).mkdir(parents=True, exist_ok=True)
            
            master_csv = f"{product_dir}/master_data.csv"
            
            # Reset index to make date a column for consistent merging
            new_data = df.copy().reset_index()
            
            # Check if master file exists
            if os.path.exists(master_csv):
                # Read existing data
                existing_data = pd.read_csv(master_csv)
                
                # Convert date column to datetime for proper comparison
                existing_data['date'] = pd.to_datetime(existing_data['date'])
                new_data['date'] = pd.to_datetime(new_data['date'])
                
                # Merge old and new data, dropping duplicates
                combined = pd.concat([existing_data, new_data]).drop_duplicates(subset=['date'])
                
                # Sort by date
                combined = combined.sort_values('date')
                
                # Save updated data
                combined.to_csv(master_csv, index=False)
                
                logger.info(f"Master data updated for {product_id}, total records: {len(combined)}")
            else:
                # Create new master file
                new_data.to_csv(master_csv, index=False)
                logger.info(f"Master data created for {product_id}, total records: {len(new_data)}")
                
        except Exception as e:
            logger.error(f"Error updating master data for {product_id}: {e}")
    
    def save_analysis_results(self):
        """
        Save the analysis results to a CSV file
        
        """
        if not self.results:
            logger.warning("No results to save")
            return
            
        try:
            # Create data directory
            base_dir = "crypto_data"
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            
            # Create filename with date range
            filename_base = f"analysis_results_{self.start_date}_to_{self.end_date}"
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.results)
            
            # Save to CSV
            csv_path = f"{base_dir}/{filename_base}.csv"
            results_df.to_csv(csv_path, index=False)
            logger.info(f"Analysis results saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            import traceback
            traceback.print_exc()
    
    def fetch_and_save_historical_data_incremental(self, product_id, end_date=None, 
                                                   periods=5, chunk_size_days=60):
        """
        Fetch historical data incrementally by going backward in chunks from current date
        
        Args:
            product_id: The product identifier (e.g., 'BTC-USD')
            end_date: End date for the most recent chunk (default: today)
            periods: Number of chunks to fetch (default: 5)
            chunk_size_days: Size of each chunk in days (default: 60)
            
        Returns:
            DataFrame with combined OHLC data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        combined_df = None
        
        for i in range(periods):
            # Calculate start and end dates for this chunk
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=(i * chunk_size_days))
            start_dt = end_dt - timedelta(days=chunk_size_days)
            
            chunk_end_date = end_dt.strftime('%Y-%m-%d')
            chunk_start_date = start_dt.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching chunk {i+1}/{periods} for {product_id}: {chunk_start_date} to {chunk_end_date}")
            
            # Store original dates to restore later
            original_start = self.start_date
            original_end = self.end_date
            original_start_timestamp = self.start_timestamp
            original_end_timestamp = self.end_timestamp
            
            # Temporarily update dates for this chunk
            self.start_date = chunk_start_date
            self.end_date = chunk_end_date
            self.start_timestamp = int(start_dt.timestamp())
            self.end_timestamp = int(end_dt.timestamp())
            
            # Re-calculate granularity for each chunk to maximize data points
            original_granularity = self.granularity
            self.granularity = self.calculate_optimal_granularity()
            
            # Fetch data for this chunk
            chunk_df = self.fetch_historical_data(product_id)
            
            # Restore original settings
            self.start_date = original_start
            self.end_date = original_end
            self.start_timestamp = original_start_timestamp
            self.end_timestamp = original_end_timestamp
            self.granularity = original_granularity
            
            if chunk_df is not None and len(chunk_df) > 0:
                # Save this chunk's data
                self.save_data_to_spreadsheet(product_id, chunk_df)
                
                # Add to combined dataframe
                if combined_df is None:
                    combined_df = chunk_df
                else:
                    combined_df = pd.concat([combined_df, chunk_df])
                    
                # Remove duplicates and sort
                combined_df = combined_df.loc[~combined_df.index.duplicated(keep='first')]
                combined_df = combined_df.sort_index()
                
            # Sleep to avoid hitting API rate limits
            time.sleep(2.0)
            
        return combined_df

    def update_combined_master_file(self):
        """
        Create or update a combined master data file containing data from all cryptocurrencies
        with a cryptocurrency identifier column for easy filtering.
        
        This is particularly useful for cross-cryptocurrency analysis.
        """
        try:
            # Create base data directory
            base_dir = "crypto_data"
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            
            combined_master_file = f"{base_dir}/all_currencies_master.csv"
            all_data = []
            
            # Get all subdirectories (one per cryptocurrency)
            crypto_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            logger.info(f"Found {len(crypto_dirs)} cryptocurrency directories")
            
            for crypto_dir in crypto_dirs:
                # Convert directory name back to product_id format
                product_id = crypto_dir.replace('_', '-')
                
                master_file = f"{base_dir}/{crypto_dir}/master_data.csv"
                if os.path.exists(master_file):
                    # Read master data for this cryptocurrency
                    df = pd.read_csv(master_file)
                    
                    # Add cryptocurrency identifier column
                    df['cryptocurrency'] = product_id
                    
                    # Add to collection
                    all_data.append(df)
                    logger.info(f"Added {len(df)} records for {product_id}")
            
            if all_data:
                # Combine all cryptocurrency data
                combined_df = pd.concat(all_data)
                
                # Ensure date is in datetime format
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                
                # Sort by cryptocurrency and date
                combined_df = combined_df.sort_values(['cryptocurrency', 'date'])
                
                # Save combined data
                combined_df.to_csv(combined_master_file, index=False)
                logger.info(f"Combined master file created with {len(combined_df)} records from {len(all_data)} cryptocurrencies")
                return combined_master_file
            else:
                logger.warning("No data found to create combined master file")
                return None
                
        except Exception as e:
            logger.error(f"Error creating combined master file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_analysis(self):
        """
        Run the analysis on all available local symbols
        
        Returns:
            list: Sorted results with the best candidates first
        """
        if self.symbols:
            symbols = self.symbols
        else:
            symbols = self.get_available_symbols()
        
        if self.test_limit > 0:
            symbols = symbols[:self.test_limit]
            
        results = []
        
        # Process each symbol
        for symbol in symbols:
            result = self.analyze_product(symbol)
            if result is not None:
                results.append(result)
                
        # Sort results by volatility (highest first)
        results.sort(key=lambda x: x['volatility'], reverse=True)
        self.results = results
        
        return results
        
    def print_results(self, limit=10):
        """
        Print the analysis results in a formatted table
        
        Args:
            limit: Maximum number of results to show
        """
        if not self.results:
            print("No cryptocurrencies met the criteria for sideways volatility")
            return
            
        # Print top results
        print(f"\n{'=' * 100}")
        print(f"CRYPTO SIDEWAYS MARKET ANALYSIS ({self.start_date} to {self.end_date})")
        print(f"{'=' * 100}")
        print(f"{'PRODUCT':<10} {'VOLATILITY':<12} {'SUPPORT':<20} {'RESISTANCE':<20} {'RANGE %':<10}")
        print(f"{'-' * 100}")
        
        for result in self.results[:limit]:
            print(f"{result['product_id']:<10} "
                  f"{result['volatility']:.2%}{'':8} "
                  f"${result['support_level']:<8.2f} ({result['support_touches']} touches){'':2} "
                  f"${result['resistance_level']:<8.2f} ({result['resistance_touches']} touches){'':2} "
                  f"{result['price_range_percent']:.2%}")
                  
        print(f"{'-' * 100}")
        
        # Print detailed analysis of the top result
        if self.results:
            best = self.results[0]
            print(f"\nBEST CANDIDATE: {best['product_id']}")
            print(f"Volatility: {best['volatility']:.2%}")
            print(f"Support level: ${best['support_level']:.2f} (touched {best['support_touches']} times)")
            print(f"Resistance level: ${best['resistance_level']:.2f} (touched {best['resistance_touches']} times)")
            print(f"Current price: ${best['last_price']:.2f}")
            print(f"Time frame analyzed: {self.start_date} to {self.end_date}")
            print(f"{'=' * 100}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze local cryptocurrency data for sideways volatility patterns')
    
    # Date range parameters
    parser.add_argument('--start', type=str, 
                      default=DEFAULT_START_DATE,
                      help=f'Start date (YYYY-MM-DD), default: {DEFAULT_DAYS_LOOKBACK} days ago')
    parser.add_argument('--end', type=str, 
                      default=DEFAULT_END_DATE,
                      help='End date (YYYY-MM-DD), default: today')
    
    # Analysis parameters                  
    parser.add_argument('--volatility', type=float, default=DEFAULT_VOLATILITY_THRESHOLD,
                      help=f'Minimum volatility threshold (default: {DEFAULT_VOLATILITY_THRESHOLD} or {DEFAULT_VOLATILITY_THRESHOLD*100}%)')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOUCH_TOLERANCE,
                      help=f'Touch tolerance for support/resistance (default: {DEFAULT_TOUCH_TOLERANCE} or {DEFAULT_TOUCH_TOLERANCE*100}%)')
    parser.add_argument('--min-touches', type=int, default=DEFAULT_MIN_TOUCHES,
                      help=f'Minimum number of touches for valid support/resistance (default: {DEFAULT_MIN_TOUCHES})')
    parser.add_argument('--limit', type=int, default=DEFAULT_RESULTS_LIMIT,
                      help=f'Maximum number of results to display (default: {DEFAULT_RESULTS_LIMIT})')
    parser.add_argument('--test-limit', type=int, default=DEFAULT_TEST_LIMIT,
                      help=f'Limit the number of cryptocurrencies to analyze for testing (default: {DEFAULT_TEST_LIMIT}, 0 for no limit)')
    parser.add_argument('--symbols', type=str, nargs="*",
                      help='Specific symbols to analyze (e.g., BTC-USDC)')
    
    return parser.parse_args()

def main():
    """Main function to run the analysis"""
    args = parse_arguments()
    
    print(f"Crypto Sideways Market Analyzer (Local Data Only)")
    print(f"Time period: {args.start or 'ALL'} to {args.end or 'ALL'}")
    
    analyzer = CryptoSidewaysAnalyzer(
        data_dir="market_data",
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        volatility_threshold=args.volatility,
        touch_tolerance=args.tolerance,
        min_touches=args.min_touches,
        test_limit=args.test_limit
    )
    
    results = analyzer.run_analysis()
    analyzer.print_results(limit=args.limit)

if __name__ == "__main__":
    main()