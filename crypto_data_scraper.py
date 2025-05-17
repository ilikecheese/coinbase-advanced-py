#!/usr/bin/env python3
"""
Crypto Data Scraper

A dedicated script to efficiently retrieve historical cryptocurrency price data from Coinbase.
This script focuses on downloading 1-minute candle data for the past 30 days by properly
chunking API calls to stay within Coinbase's rate limits and 350-candle per request limit.

Features:
- Downloads 1-minute candles for specified cryptocurrencies
- Handles Coinbase API's 350-candle limit by breaking requests into appropriate chunks
- Saves data in both individual files per crypto and in a combined master file
- Implements proper error handling and retry logic
- Provides detailed logging about the data collection process
"""

# === USER CONFIGURABLE OPTIONS ===
DAYS_LOOKBACK = 30         # Number of days of historical data to retrieve
QUOTE_CURRENCY = "USDC"    # Quote currency (e.g., USDC, USD)
SYMBOLS = ["FLOKI-USDC"]             # List of symbols to scrape (e.g., ["BTC-USDC", "ETH-USDC"]), or None for all
LIMIT = None               # Limit the number of symbols to scrape (int or None)
# ================================

import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import traceback
from pathlib import Path
from coinbase.rest import RESTClient
from config import API_KEY, API_SECRET
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('crypto_data_scraper')

class CryptoDataScraper:
    """
    Class for scraping historical cryptocurrency price data from Coinbase
    with appropriate chunking to handle API limits
    """
    
    # Define constants
    MAX_CANDLES_PER_REQUEST = 300  # Use a slightly lower value than the 350 limit to be safe
    BASE_DATA_DIR = "market_data"  # Directory to store all scraped data
    GRANULARITY = "ONE_MINUTE"     # 1-minute candles
    
    def __init__(self, api_key, api_secret, days_lookback=30, quote_currency="USDC"):
        """
        Initialize the scraper
        
        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
            days_lookback: Number of days of historical data to retrieve
            quote_currency: Base quote currency to use (e.g., "USDC", "USD")
        """
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.days_lookback = days_lookback
        self.quote_currency = quote_currency
        
        # Calculate time range
        self.end_time = datetime.now()
        self.start_time = self.end_time - timedelta(days=days_lookback)
        
        # Create data directory if it doesn't exist
        Path(self.BASE_DATA_DIR).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CryptoDataScraper initialized to collect {days_lookback} days of ONE_MINUTE candles")
        logger.info(f"Time range: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} to {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def fetch_products(self):
        """
        Fetch all available cryptocurrency trading pairs from Coinbase
        
        Returns:
            list: All available trading pairs for the specified quote currency
        """
        try:
            products = self.client.get_public_products(product_type="SPOT")
            
            # Filter for the specified quote currency
            quote_products = [p for p in products.products if p.product_id.endswith(f'-{self.quote_currency}')]
            
            logger.info(f"Found {len(quote_products)} {self.quote_currency} trading pairs")
            
            # If no products found with the specified quote currency, try falling back to USD
            if not quote_products and self.quote_currency != "USD":
                logger.info(f"No {self.quote_currency} pairs found. Falling back to USD pairs.")
                usd_products = [p for p in products.products if p.product_id.endswith('-USD')]
                logger.info(f"Found {len(usd_products)} USD trading pairs")
                return usd_products
                
            return quote_products
        except Exception as e:
            logger.error(f"Error fetching products: {e}")
            return []
    
    def fetch_historical_data(self, product_id):
        """
        Fetch historical 1-minute candle data for a specific product,
        breaking requests into chunks to stay within API limits
        
        Args:
            product_id: The product identifier (e.g., 'BTC-USDC')
            
        Returns:
            DataFrame with OHLC data or None if error
        """
        try:
            logger.info(f"Fetching historical data for {product_id}")
            
            # Calculate total time range in seconds
            start_timestamp = int(self.start_time.timestamp())
            end_timestamp = int(self.end_time.timestamp())
            time_diff_seconds = end_timestamp - start_timestamp
            
            # Each 1-minute candle represents 60 seconds
            seconds_per_candle = 60
            
            # Calculate maximum seconds per API call to stay within candle limit
            max_seconds_per_call = self.MAX_CANDLES_PER_REQUEST * seconds_per_candle
            
            # Calculate how many API calls we need to make
            chunks_needed = int(np.ceil(time_diff_seconds / max_seconds_per_call))
            logger.info(f"Need to make {chunks_needed} API calls to fetch all data for {product_id}")
            
            # Get data in chunks
            all_data = []
            for i in range(chunks_needed):
                chunk_start = start_timestamp + i * max_seconds_per_call
                chunk_end = min(end_timestamp, start_timestamp + (i + 1) * max_seconds_per_call)
                
                chunk_start_date = datetime.fromtimestamp(chunk_start).strftime('%Y-%m-%d %H:%M:%S')
                chunk_end_date = datetime.fromtimestamp(chunk_end).strftime('%Y-%m-%d %H:%M:%S')
                
                # Progress indicator
                logger.info(f"Fetching chunk {i+1}/{chunks_needed} for {product_id}: {chunk_start_date} to {chunk_end_date}")
                
                # Add delay between API calls to avoid rate limiting
                if i > 0:
                    time.sleep(1.0)
                
                # Get candles data from Coinbase for this chunk
                try:
                    candles_response = self.client.get_public_candles(
                        product_id=product_id,
                        start=str(chunk_start),
                        end=str(chunk_end),
                        granularity=self.GRANULARITY
                    )
                    
                    if hasattr(candles_response, 'candles') and candles_response.candles:
                        chunk_data = []
                        for candle in candles_response.candles:
                            chunk_data.append({
                                'timestamp': int(candle.start),
                                'open': float(candle.open),
                                'high': float(candle.high),
                                'low': float(candle.low),
                                'close': float(candle.close),
                                'volume': float(candle.volume)
                            })
                        all_data.extend(chunk_data)
                        logger.info(f"Fetched {len(chunk_data)} candles for chunk {i+1}")
                    else:
                        logger.warning(f"No data received for chunk {i+1}")
                except Exception as chunk_error:
                    logger.error(f"Error fetching chunk {i+1} for {product_id}: {chunk_error}")
                    
                    # If we get an error, wait a bit longer before retrying
                    time.sleep(5.0)
                    
                    # Try once more
                    try:
                        candles_response = self.client.get_public_candles(
                            product_id=product_id,
                            start=str(chunk_start),
                            end=str(chunk_end),
                            granularity=self.GRANULARITY
                        )
                        
                        if hasattr(candles_response, 'candles') and candles_response.candles:
                            chunk_data = []
                            for candle in candles_response.candles:
                                chunk_data.append({
                                    'timestamp': int(candle.start),
                                    'open': float(candle.open),
                                    'high': float(candle.high),
                                    'low': float(candle.low),
                                    'close': float(candle.close),
                                    'volume': float(candle.volume)
                                })
                            all_data.extend(chunk_data)
                            logger.info(f"Retry successful! Fetched {len(chunk_data)} candles for chunk {i+1}")
                        else:
                            logger.warning(f"Retry failed: No data received for chunk {i+1}")
                    except Exception as retry_error:
                        logger.error(f"Retry failed for chunk {i+1}: {retry_error}")
            
            if not all_data:
                logger.warning(f"No candle data available for {product_id} across all chunks")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Set index to date
            df = df.set_index('date')
            df = df.sort_index()  # Ensure chronological order
            
            # Remove duplicates that might occur at chunk boundaries
            df = df[~df.index.duplicated(keep='first')]
            
            # Verify the time interval between candles
            if len(df) > 1:
                interval_check = []
                for i in range(1, min(10, len(df))):
                    diff_seconds = (df.index[i] - df.index[i-1]).total_seconds()
                    interval_check.append(diff_seconds)
                
                avg_interval = sum(interval_check) / len(interval_check)
                logger.info(f"Average interval between candles for {product_id}: {avg_interval:.1f} seconds")
                
                if abs(avg_interval - 60) > 10:  # More than 10 seconds off from expected 60 seconds
                    logger.warning(f"WARNING: Expected 1-minute candles but average interval is {avg_interval:.1f} seconds")
            
            logger.info(f"Successfully fetched {len(df)} one-minute candles for {product_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {product_id}: {e}")
            traceback.print_exc()
            return None
    
    def save_data(self, product_id, df):
        """
        Save the fetched data in both individual files and update a master file
        
        Args:
            product_id: The product identifier (e.g., 'BTC-USDC')
            df: DataFrame with OHLC data
            
        Returns:
            str: Path to saved file or None if error
        """
        if df is None or len(df) < 1:
            logger.warning(f"No data to save for {product_id}")
            return None
            
        try:
            # Create directory structure
            symbol_dir = f"{self.BASE_DATA_DIR}/{product_id.replace('-', '_')}"
            Path(symbol_dir).mkdir(parents=True, exist_ok=True)
            
            # Format date range for filename
            start_date = df.index.min().strftime('%Y%m%d')
            end_date = df.index.max().strftime('%Y%m%d')
            
            # Format the filename - use CSV instead of Parquet
            filename = f"{product_id.replace('-', '_')}_{start_date}_to_{end_date}_{self.GRANULARITY.lower()}.csv"
            file_path = f"{symbol_dir}/{filename}"
            
            # Reset index to include date as a column
            df_save = df.copy().reset_index()
            
            # Save to CSV
            df_save.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}")
            
            # Update master file for this symbol
            self._update_master_file(product_id, df)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving data for {product_id}: {e}")
            traceback.print_exc()
            return None
    
    def _update_master_file(self, product_id, df):
        """
        Update the master file for a specific product
        
        Args:
            product_id: The product identifier (e.g., 'BTC-USDC')
            df: DataFrame with new data to append
        """
        if df is None or len(df) < 1:
            return
            
        try:
            # Directory and filename
            symbol_dir = f"{self.BASE_DATA_DIR}/{product_id.replace('-', '_')}"
            master_file = f"{symbol_dir}/master_data.csv"
            
            # Reset index to include date as a column for consistent merging
            new_data = df.copy().reset_index()
            
            # Check if master file exists
            if os.path.exists(master_file):
                # Read existing data
                existing_data = pd.read_csv(master_file)
                
                # Ensure date columns are datetime for proper comparison
                existing_data['date'] = pd.to_datetime(existing_data['date'])
                new_data['date'] = pd.to_datetime(new_data['date'])
                
                # Merge old and new data, dropping duplicates
                combined = pd.concat([existing_data, new_data]).drop_duplicates(subset=['date'])
                
                # Sort by date
                combined = combined.sort_values('date')
                
                # Save updated data
                combined.to_csv(master_file, index=False)
                
                logger.info(f"Master data updated for {product_id}, total records: {len(combined)}")
            else:
                # Create new master file
                new_data.to_csv(master_file, index=False)
                logger.info(f"Master data created for {product_id}, total records: {len(new_data)}")
                
        except Exception as e:
            logger.error(f"Error updating master data for {product_id}: {e}")
            traceback.print_exc()
    
    def create_combined_master_file(self):
        """
        Create a combined master file containing all cryptocurrency data
        with an additional column for the cryptocurrency identifier
        
        Returns:
            str: Path to combined master file or None if error
        """
        try:
            combined_master_file = f"{self.BASE_DATA_DIR}/all_crypto_master.csv"
            all_data = []
            
            # Get all subdirectories (one per cryptocurrency)
            crypto_dirs = [d for d in os.listdir(self.BASE_DATA_DIR) 
                         if os.path.isdir(os.path.join(self.BASE_DATA_DIR, d))]
            
            if not crypto_dirs:
                logger.warning("No cryptocurrency data directories found")
                return None
                
            logger.info(f"Found {len(crypto_dirs)} cryptocurrency directories")
            
            for crypto_dir in crypto_dirs:
                # Convert directory name back to product_id format
                product_id = crypto_dir.replace('_', '-')
                
                # Path to master file for this cryptocurrency
                master_file = f"{self.BASE_DATA_DIR}/{crypto_dir}/master_data.csv"
                
                if os.path.exists(master_file):
                    # Read master data
                    df = pd.read_csv(master_file)
                    
                    # Add product identifier column
                    df['symbol'] = product_id
                    
                    # Add to collection
                    all_data.append(df)
                    logger.info(f"Added {len(df)} records for {product_id}")
            
            if all_data:
                # Combine all data
                combined_df = pd.concat(all_data)
                
                # Ensure date is datetime
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                
                # Sort by symbol and date
                combined_df = combined_df.sort_values(['symbol', 'date'])
                
                # Save combined data
                combined_df.to_csv(combined_master_file, index=False)
                
                logger.info(f"Combined master file created at {combined_master_file}")
                logger.info(f"Total records: {len(combined_df)}, Symbols: {len(all_data)}")
                
                return combined_master_file
            else:
                logger.warning("No data found to create combined master file")
                return None
                
        except Exception as e:
            logger.error(f"Error creating combined master file: {e}")
            traceback.print_exc()
            return None
    
    def run(self, symbols=None, limit=None):
        """
        Run the data scraper for specified symbols or all available symbols
        
        Args:
            symbols: List of specific symbols to scrape (e.g., ['BTC-USDC', 'ETH-USDC'])
                    If None, all available symbols will be scraped
            limit: Maximum number of symbols to scrape (for testing)
            
        Returns:
            list: Paths to saved data files
        """
        saved_files = []
        
        # Get symbols to scrape
        if symbols:
            products_to_scrape = symbols
            logger.info(f"Will scrape data for {len(symbols)} specified symbols")
        else:
            # Fetch all available products
            products = self.fetch_products()
            if not products:
                logger.error("No products found to scrape")
                return []
                
            products_to_scrape = [p.product_id for p in products]
            logger.info(f"Found {len(products_to_scrape)} products to scrape")
            
        # Apply limit if specified
        if limit and limit > 0:
            products_to_scrape = products_to_scrape[:limit]
            logger.info(f"Limited to {limit} products for scraping")
        
        # Process each product
        for i, product_id in enumerate(products_to_scrape):
            logger.info(f"Processing {i+1}/{len(products_to_scrape)}: {product_id}")
            
            # Skip stablecoins as they typically have low volatility
            if any(stablecoin in product_id for stablecoin in ['USDC-', 'USDT-', 'DAI-', 'BUSD-']):
                logger.info(f"Skipping stablecoin {product_id}")
                continue
                
            # Fetch and save historical data
            df = self.fetch_historical_data(product_id)
            if df is not None and len(df) > 0:
                file_path = self.save_data(product_id, df)
                if file_path:
                    saved_files.append(file_path)
                    
            # Sleep briefly to avoid hitting API rate limits
            time.sleep(1.5)
        
        # Create combined master file
        combined_master = self.create_combined_master_file()
        if combined_master:
            saved_files.append(combined_master)
        
        logger.info(f"Data scraping completed. Saved {len(saved_files)} files.")
        return saved_files

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Crypto Data Scraper")
    parser.add_argument("--days", type=int, default=None, help="Number of days of historical data to retrieve")
    parser.add_argument("--quote", type=str, default=None, help="Quote currency (e.g., USDC, USD)")
    parser.add_argument("--symbols", type=str, nargs="*", default=None, help="Specific symbols to scrape (e.g., BTC-USDC)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of symbols to scrape")
    return parser.parse_args()

def main():
    """Main entry point for the script"""
    args = parse_arguments()

    # Use CLI args if provided, otherwise fall back to constants at the top
    days = args.days if args.days is not None else DAYS_LOOKBACK
    quote = args.quote if args.quote is not None else QUOTE_CURRENCY
    symbols = args.symbols if args.symbols is not None else SYMBOLS
    limit = args.limit if args.limit is not None else LIMIT

    print(f"Crypto Data Scraper - 1-minute Candles")
    print(f"Retrieving {days} days of historical data")
    print(f"Quote currency: {quote}")

    # Initialize and run the scraper
    try:
        scraper = CryptoDataScraper(
            api_key=API_KEY,
            api_secret=API_SECRET,
            days_lookback=days,
            quote_currency=quote
        )

        saved_files = scraper.run(symbols=symbols, limit=limit)

        if saved_files:
            print(f"\nData scraping completed successfully!")
            print(f"Data saved to {scraper.BASE_DATA_DIR}/ directory")
            print(f"Total files saved: {len(saved_files)}")

            # Print path to combined master file if it was created
            master_path = f"{scraper.BASE_DATA_DIR}/all_crypto_master.csv"
            if os.path.exists(master_path):
                print(f"\nCombined master file: {master_path}")
        else:
            print("\nNo data files were saved. Check the logs for errors.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()