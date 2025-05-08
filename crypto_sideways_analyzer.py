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
"""

import os
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import signal
from scipy.cluster.vq import kmeans
from coinbase.rest import RESTClient
from config import API_KEY, API_SECRET

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crypto_sideways_analyzer')

class CryptoSidewaysAnalyzer:
    """Analyzes cryptocurrencies for sideways market patterns with volatility"""
    
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
    
    def __init__(self, api_key, api_secret, start_date, end_date, 
                volatility_threshold=0.02, sideways_tolerance=0.1, 
                touch_tolerance=0.01, min_touches=3, test_limit=0, granularity=None):
        """
        Initialize the analyzer with API credentials and parameters
        
        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            volatility_threshold: Minimum volatility (std dev of returns) to consider
            sideways_tolerance: Maximum variance in price range to qualify as sideways
            touch_tolerance: Percentage tolerance for price "touches" on support/resistance
            min_touches: Minimum number of touches to consider valid support/resistance
            test_limit: Limit the number of cryptocurrencies to analyze for testing
            granularity: Time interval for candles (if None, will auto-calculate based on date range)
        """
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.start_date = start_date
        self.end_date = end_date
        self.volatility_threshold = volatility_threshold
        self.sideways_tolerance = sideways_tolerance
        self.touch_tolerance = touch_tolerance
        self.min_touches = min_touches
        self.test_limit = test_limit
        self.results = []
        
        # Convert dates to Unix timestamps (Coinbase API format)
        self.start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        self.end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        
        # Auto-select granularity if not specified
        if granularity is None:
            self.granularity = self.calculate_optimal_granularity()
        else:
            self.granularity = granularity
            
        logger.info(f"Using {self.granularity} granularity for time period {start_date} to {end_date}")
    
    def calculate_optimal_granularity(self):
        """
        Calculate the optimal granularity based on the date range to stay under 350 candles
        
        Returns:
            str: The optimal granularity option
        """
        # Calculate the time difference in minutes
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        time_diff_minutes = (end_dt - start_dt).total_seconds() / 60
        
        # If we have less than 350 minutes, use ONE_MINUTE
        if time_diff_minutes <= self.MAX_CANDLES:
            return "ONE_MINUTE"
            
        # Calculate the minimum granularity needed
        min_granularity_minutes = int(time_diff_minutes / self.MAX_CANDLES) + 1
        
        # Find the smallest granularity option that fits
        for option, minutes in sorted(self.GRANULARITY_OPTIONS.items(), key=lambda x: x[1]):
            if minutes >= min_granularity_minutes:
                return option
                
        # If no option fits within 350 candles, use largest available (ONE_DAY)
        return "ONE_DAY"
        
    def fetch_products(self):
        """Fetch all available cryptocurrency products from Coinbase"""
        try:
            products = self.client.get_public_products(product_type="SPOT")
            # Filter for USD pairs only
            usd_products = [p for p in products.products if p.product_id.endswith('-USD')]
            logger.info(f"Found {len(usd_products)} USD trading pairs")
            return usd_products
        except Exception as e:
            logger.error(f"Error fetching products: {e}")
            return []
            
    def fetch_historical_data(self, product_id):
        """
        Fetch historical OHLC data for a specific product
        
        Args:
            product_id: The product identifier (e.g., 'BTC-USD')
            
        Returns:
            DataFrame with OHLC data or None if error
        """
        try:
            logger.info(f"Fetching historical data for {product_id}")
            
            # Coinbase API has rate limits, so add a small delay between requests
            time.sleep(1.0)  # Increased delay to avoid rate limiting
            
            # Get candles data from Coinbase
            candles_response = self.client.get_public_candles(
                product_id=product_id,
                start=str(self.start_timestamp),
                end=str(self.end_timestamp),
                granularity=self.granularity  # Use the granularity parameter
            )
            
            if not hasattr(candles_response, 'candles') or not candles_response.candles:
                logger.warning(f"No candle data available for {product_id}")
                return None
                
            # Convert to DataFrame
            data = []
            for candle in candles_response.candles:
                data.append({
                    'timestamp': int(candle.start),
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': float(candle.volume)
                })
                
            if not data:
                return None
                
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('date')
            df = df.sort_index()  # Ensure chronological order
            
            # Calculate daily returns for volatility analysis
            df['return'] = df['close'].pct_change()
            
            logger.info(f"Fetched {len(df)} data points for {product_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {product_id}: {e}")
            return None
            
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
        
    def analyze_product(self, product_id):
        """
        Analyze a single product for sideways volatility pattern
        
        Args:
            product_id: The product to analyze (e.g., 'BTC-USD')
            
        Returns:
            dict: Analysis results or None if criteria not met
        """
        # Fetch historical data
        df = self.fetch_historical_data(product_id)
        if df is None or len(df) < 30:
            return None
            
        # Calculate volatility
        volatility = self.calculate_volatility(df)
        
        # Check if volatility meets threshold
        if volatility < self.volatility_threshold:
            logger.info(f"{product_id} volatility {volatility:.2%} below threshold {self.volatility_threshold:.2%}")
            return None
            
        # Check if market is sideways
        is_sideways, sideways_details = self.is_sideways_market(df)
        
        if not is_sideways:
            logger.info(f"{product_id} is not in a sideways market")
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
            logger.info(f"{product_id} doesn't have strong support/resistance levels")
            return None
            
        # Get the best support and resistance levels (most touches)
        best_support = max(valid_support, key=lambda x: x[1]) if valid_support else (0, 0)
        best_resistance = max(valid_resistance, key=lambda x: x[1]) if valid_resistance else (0, 0)
        
        # Calculate range as percentage
        if best_support[0] > 0:
            range_percent = (best_resistance[0] - best_support[0]) / best_support[0]
        else:
            range_percent = 0
            
        # Prepare results
        result = {
            'product_id': product_id,
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
        
        logger.info(f"Found candidate: {product_id} - Volatility: {volatility:.2%}, "
                   f"Support: ${best_support[0]:.2f} ({best_support[1]} touches), "
                   f"Resistance: ${best_resistance[0]:.2f} ({best_resistance[1]} touches)")
                   
        return result

    def run_analysis(self):
        """
        Run the analysis on all USD cryptocurrency pairs
        
        Returns:
            list: Sorted results with the best candidates first
        """
        # Get all products
        products = self.fetch_products()
        
        if not products:
            logger.error("No products found to analyze")
            return []
            
        results = []
        
        # Process each product
        for i, product in enumerate(products):
            if self.test_limit > 0 and i >= self.test_limit:
                break
            
            product_id = product.product_id
            
            # Skip stablecoins as they usually have very low volatility
            if any(stablecoin in product_id for stablecoin in ['USDC-', 'USDT-', 'DAI-', 'BUSD-']):
                continue
                
            result = self.analyze_product(product_id)
            if result is not None:
                results.append(result)
                
            # Sleep briefly to avoid hitting API rate limits
            time.sleep(1.0)  # Increased delay to prevent rate limits
            
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
    parser = argparse.ArgumentParser(description='Analyze cryptocurrencies for sideways volatility patterns')
    
    # Date range parameters
    parser.add_argument('--start', type=str, 
                      default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                      help='Start date (YYYY-MM-DD), default: 1 year ago')
    parser.add_argument('--end', type=str, 
                      default=datetime.now().strftime('%Y-%m-%d'),
                      help='End date (YYYY-MM-DD), default: today')
    
    # Analysis parameters                  
    parser.add_argument('--volatility', type=float, default=0.02,
                      help='Minimum volatility threshold (default: 0.02 or 2%)')
    parser.add_argument('--tolerance', type=float, default=0.01,
                      help='Touch tolerance for support/resistance (default: 0.01 or 1%)')
    parser.add_argument('--min-touches', type=int, default=3,
                      help='Minimum number of touches for valid support/resistance (default: 3)')
    parser.add_argument('--limit', type=int, default=5,
                      help='Maximum number of results to display (default: 5)')
    parser.add_argument('--test-limit', type=int, default=10,
                      help='Limit the number of cryptocurrencies to analyze for testing (default: 10, 0 for no limit)')
    parser.add_argument('--granularity', type=str, default=None, choices=[
                      "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE",
                      "ONE_HOUR", "TWO_HOUR", "SIX_HOUR", "ONE_DAY"],
                      help='Candle time interval (default: auto-calculated based on date range)')
    
    return parser.parse_args()

def main():
    """Main function to run the analysis"""
    args = parse_arguments()
    
    print(f"Crypto Sideways Market Analyzer")
    print(f"Time period: {args.start} to {args.end}")
    print(f"Analyzing Coinbase cryptocurrencies for sideways volatility patterns...")
    
    try:
        # Create analyzer with parameters from arguments
        analyzer = CryptoSidewaysAnalyzer(
            api_key=API_KEY,
            api_secret=API_SECRET,
            start_date=args.start,
            end_date=args.end,
            volatility_threshold=args.volatility,
            touch_tolerance=args.tolerance,
            min_touches=args.min_touches,
            test_limit=args.test_limit,
            granularity=args.granularity
        )
        
        # Run the analysis
        results = analyzer.run_analysis()
        
        # Print results
        analyzer.print_results(limit=args.limit)
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()