#!/usr/bin/env python3
"""
Crypto Data Quality Check

This script performs comprehensive quality checks on cryptocurrency market data to ensure it meets
the standards required for algorithmic trading and analysis.

Key quality checks:
1. Data completeness and continuity - Identifies gaps in time series data
2. Price accuracy and outlier detection - Flags suspicious price movements
3. OHLC consistency verification - Ensures candle data follows logical rules
4. Volume analysis - Detects unusual trading volume patterns  
5. Cross-check capabilities - Framework for comparing with other data sources

Why this matters:
- Trading algorithms are extremely sensitive to data quality issues
- Missing candles can lead to false signals and trading losses
- Price anomalies might indicate erroneous data rather than actionable movements
- Consistent verification ensures your backtests reflect real-world possibilities

Usage:
    python data_quality_check.py --dir market_data [--crypto BTC-USDC] [--report report.html]
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_quality.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_quality_check')

class DataQualityChecker:
    """
    Class for checking data quality in cryptocurrency market data
    """
    
    def __init__(self, data_dir="market_data"):
        """
        Initialize the data quality checker
        
        Args:
            data_dir: Directory containing cryptocurrency market data
        """
        self.data_dir = data_dir
        self.results = {}
        logger.info(f"Initializing data quality checks on {data_dir}")
        
    def check_data_continuity(self, df, symbol, expected_interval_seconds=60):
        """
        Check for gaps in time series data
        
        Args:
            df: DataFrame containing market data
            symbol: Cryptocurrency symbol
            expected_interval_seconds: Expected time between candles (default: 60 for 1-min candles)
            
        Returns:
            DataFrame of gaps found in the data
        """
        logger.info(f"Checking data continuity for {symbol}")
        
        # Ensure datetime format
        if 'date' not in df.columns:
            logger.error(f"No 'date' column found in {symbol} data")
            return pd.DataFrame()
            
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        df = df.sort_values('date')
        
        # Check time difference between consecutive rows
        df['time_diff'] = df['date'].diff().dt.total_seconds()
        
        # For 1-minute data, gaps are where diff > expected_interval_seconds
        gaps = df[df['time_diff'] > expected_interval_seconds * 1.5]
        
        # Calculate completeness percentage
        expected_intervals = int((df['date'].max() - df['date'].min()).total_seconds() / expected_interval_seconds)
        actual_intervals = len(df)
        completeness = (actual_intervals / expected_intervals) * 100 if expected_intervals > 0 else 0
        
        logger.info(f"{symbol}: Found {len(gaps)} gaps in the data")
        logger.info(f"{symbol}: Data completeness: {completeness:.2f}%")
        
        # Store results for this symbol
        self.results[symbol] = self.results.get(symbol, {})
        self.results[symbol]['gaps_count'] = len(gaps)
        self.results[symbol]['completeness_pct'] = completeness
        
        return gaps
        
    def detect_price_anomalies(self, df, symbol, std_threshold=3):
        """
        Identify suspicious price movements and outliers
        
        Args:
            df: DataFrame containing market data
            symbol: Cryptocurrency symbol
            std_threshold: Number of standard deviations to consider an outlier
            
        Returns:
            tuple: (DataFrame of price anomalies, DataFrame of invalid values)
        """
        logger.info(f"Checking price anomalies for {symbol}")
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            logger.error(f"Missing required OHLC columns in {symbol} data")
            return pd.DataFrame(), pd.DataFrame()
            
        df = df.copy()
        
        # Calculate percentage price changes
        df['pct_change'] = df['close'].pct_change() * 100
        
        # Identify outliers (std_threshold standard deviations from mean)
        mean_change = df['pct_change'].mean()
        std_change = df['pct_change'].std()
        
        outliers = df[(df['pct_change'] > mean_change + std_threshold*std_change) | 
                      (df['pct_change'] < mean_change - std_threshold*std_change)]
        
        # Check for zero or negative values in OHLCV
        invalid_values = df[(df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)]
        
        logger.info(f"{symbol}: Detected {len(outliers)} potential price anomalies")
        logger.info(f"{symbol}: Found {len(invalid_values)} candles with invalid price values")
        
        # Store results
        self.results[symbol] = self.results.get(symbol, {})
        self.results[symbol]['anomalies_count'] = len(outliers)
        self.results[symbol]['invalid_values_count'] = len(invalid_values)
        
        return outliers, invalid_values
    
    def verify_ohlc_consistency(self, df, symbol):
        """
        Verify that OHLC values are consistent with candle logic
        
        Args:
            df: DataFrame containing market data
            symbol: Cryptocurrency symbol
            
        Returns:
            tuple: (DataFrame of OHLC inconsistencies, DataFrame of open/previous close mismatches)
        """
        logger.info(f"Verifying OHLC consistency for {symbol}")
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            logger.error(f"Missing required OHLC columns in {symbol} data")
            return pd.DataFrame(), pd.DataFrame()
            
        df = df.copy()
        
        # Check if high >= open, close, low and low <= open, close
        inconsistencies = df[
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) | 
            (df['low'] > df['open']) | 
            (df['low'] > df['close'])
        ]
        
        # Check if current open == previous close
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
                
            df = df.sort_values('date')
            df['prev_close'] = df['close'].shift(1)
            open_close_mismatch = df[(abs(df['open'] - df['prev_close']) > df['prev_close'] * 0.001) & 
                                     (~df['prev_close'].isna())]
        else:
            open_close_mismatch = pd.DataFrame()
        
        logger.info(f"{symbol}: Found {len(inconsistencies)} OHLC inconsistencies")
        logger.info(f"{symbol}: Found {len(open_close_mismatch)} open/previous close mismatches")
        
        # Store results
        self.results[symbol] = self.results.get(symbol, {})
        self.results[symbol]['inconsistencies_count'] = len(inconsistencies)
        self.results[symbol]['open_close_mismatch_count'] = len(open_close_mismatch)
        
        return inconsistencies, open_close_mismatch
    
    def analyze_volume_patterns(self, df, symbol, std_threshold=3):
        """
        Analyze trading volume for suspicious patterns
        
        Args:
            df: DataFrame containing market data
            symbol: Cryptocurrency symbol
            std_threshold: Number of standard deviations to consider an outlier
            
        Returns:
            tuple: (DataFrame of zero volume candles, DataFrame of volume outliers)
        """
        logger.info(f"Analyzing volume patterns for {symbol}")
        
        if 'volume' not in df.columns:
            logger.error(f"No volume column found in {symbol} data")
            return pd.DataFrame(), pd.DataFrame()
            
        df = df.copy()
        
        # Check for zero volume candles
        zero_volume = df[df['volume'] == 0]
        
        # Check for volume outliers
        df['volume_zscore'] = (df['volume'] - df['volume'].mean()) / df['volume'].std() if df['volume'].std() > 0 else 0
        volume_outliers = df[abs(df['volume_zscore']) > std_threshold]
        
        logger.info(f"{symbol}: Found {len(zero_volume)} candles with zero volume")
        logger.info(f"{symbol}: Found {len(volume_outliers)} volume outliers")
        
        # Store results
        self.results[symbol] = self.results.get(symbol, {})
        self.results[symbol]['zero_volume_count'] = len(zero_volume)
        self.results[symbol]['volume_outliers_count'] = len(volume_outliers)
        
        return zero_volume, volume_outliers

    def compare_with_other_source(self, our_df, other_df, symbol):
        """
        Compare our data with another source
        
        Args:
            our_df: DataFrame containing our market data
            other_df: DataFrame from another source
            symbol: Cryptocurrency symbol
            
        Returns:
            tuple: (Merged DataFrame, DataFrame of significant discrepancies)
        """
        logger.info(f"Comparing {symbol} data with external source")
        
        our_df = our_df.copy()
        other_df = other_df.copy()
        
        # Ensure datetime format is consistent
        if 'date' in our_df.columns and 'date' in other_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(our_df['date']):
                our_df['date'] = pd.to_datetime(our_df['date'])
                
            if not pd.api.types.is_datetime64_any_dtype(other_df['date']):
                other_df['date'] = pd.to_datetime(other_df['date'])
                
            # Merge on timestamp
            merged = pd.merge(our_df, other_df, on='date', suffixes=('_our', '_other'))
            
            # Calculate price differences
            if 'close_our' in merged.columns and 'close_other' in merged.columns:
                merged['close_diff_pct'] = abs(merged['close_our'] - merged['close_other']) / merged['close_our'] * 100
                
                # Identify significant discrepancies (>0.5%)
                discrepancies = merged[merged['close_diff_pct'] > 0.5]
                
                logger.info(f"{symbol}: Found {len(discrepancies)} significant price discrepancies between sources")
                logger.info(f"{symbol}: Average price difference: {merged['close_diff_pct'].mean():.4f}%")
                
                # Store results
                self.results[symbol] = self.results.get(symbol, {})
                self.results[symbol]['discrepancies_count'] = len(discrepancies)
                self.results[symbol]['avg_price_diff_pct'] = merged['close_diff_pct'].mean()
                
                return merged, discrepancies
                
        logger.error(f"Could not compare {symbol} data with external source")
        return pd.DataFrame(), pd.DataFrame()

    def generate_visualizations(self, df, symbol, output_dir="data_quality_reports"):
        """
        Generate visualizations for data quality analysis
        
        Args:
            df: DataFrame containing market data
            symbol: Cryptocurrency symbol
            output_dir: Directory to save visualizations
            
        Returns:
            list: Paths to generated visualization files
        """
        logger.info(f"Generating visualizations for {symbol}")
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        visualization_paths = []
        
        try:
            df = df.copy()
            if 'date' not in df.columns:
                logger.error(f"No 'date' column found in {symbol} data")
                return []
                
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
                
            df = df.sort_values('date')
            
            # 1. Price and volume chart
            if all(col in df.columns for col in ['close', 'volume']):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
                
                # Price chart
                ax1.plot(df['date'], df['close'], label='Close Price')
                ax1.set_title(f"{symbol} Price Chart")
                ax1.set_ylabel('Price')
                ax1.grid(True)
                
                # Volume chart
                ax2.bar(df['date'], df['volume'], label='Volume')
                ax2.set_title(f"{symbol} Volume Chart")
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Volume')
                ax2.grid(True)
                
                plt.tight_layout()
                path = f"{output_dir}/{symbol.replace('-', '_')}_price_volume.png"
                plt.savefig(path)
                plt.close(fig)
                visualization_paths.append(path)
                
            # 2. Price change distribution
            if 'close' in df.columns:
                df['pct_change'] = df['close'].pct_change() * 100
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['pct_change'].dropna(), kde=True, ax=ax)
                
                ax.set_title(f"{symbol} Percentage Price Change Distribution")
                ax.set_xlabel('Percent Change (%)')
                ax.axvline(x=0, color='red', linestyle='--')
                
                path = f"{output_dir}/{symbol.replace('-', '_')}_pct_change_dist.png"
                plt.savefig(path)
                plt.close(fig)
                visualization_paths.append(path)
                
            # 3. Time difference between candles
            df['time_diff'] = df['date'].diff().dt.total_seconds()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['time_diff'].dropna(), bins=20, ax=ax)
            
            ax.set_title(f"{symbol} Time Between Candles Distribution")
            ax.set_xlabel('Seconds')
            expected_interval = df['time_diff'].median()
            ax.axvline(x=expected_interval, color='green', linestyle='--', 
                      label=f'Expected ({expected_interval} sec)')
            ax.legend()
            
            path = f"{output_dir}/{symbol.replace('-', '_')}_time_diff.png"
            plt.savefig(path)
            plt.close(fig)
            visualization_paths.append(path)
            
            logger.info(f"{symbol}: Generated {len(visualization_paths)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            
        return visualization_paths
        
    def generate_html_report(self, output_path="data_quality_report.html"):
        """
        Generate an HTML report from the data quality check results
        
        Args:
            output_path: Path to save the HTML report
            
        Returns:
            str: Path to the generated HTML report
        """
        logger.info("Generating HTML report")
        
        if not self.results:
            logger.warning("No results to generate report from")
            return None
            
        try:
            # Convert results to DataFrame
            results_df = pd.DataFrame.from_dict(self.results, orient='index')
            
            # Create HTML
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Crypto Market Data Quality Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .good { color: green; }
                    .warning { color: orange; }
                    .error { color: red; }
                    .summary { margin: 20px 0; padding: 15px; background-color: #f0f0f0; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>Crypto Market Data Quality Report</h1>
                <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Analyzed data for """ + str(len(self.results)) + """ cryptocurrencies</p>
                </div>
                
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Cryptocurrency</th>
            """
            
            # Add column headers
            for col in results_df.columns:
                html += f"<th>{col}</th>\n"
                
            html += "</tr>\n"
            
            # Add rows
            for symbol, row in results_df.iterrows():
                html += f"<tr>\n<td>{symbol}</td>\n"
                
                for col, value in row.items():
                    # Determine color based on the metric
                    css_class = "good"
                    if col.endswith('_count') and value > 0:
                        css_class = "warning" if value < 10 else "error"
                        
                    if col == 'completeness_pct' and value < 99:
                        css_class = "warning" if value >= 95 else "error"
                        
                    # Format value
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                        if col.endswith('_pct'):
                            formatted_value += "%"
                    else:
                        formatted_value = str(value)
                        
                    html += f"<td class='{css_class}'>{formatted_value}</td>\n"
                    
                html += "</tr>\n"
                
            html += """
                </table>
                
                <div class="summary">
                    <h2>Recommendations</h2>
                    <ul>
                        <li>Address gaps in data continuity for any cryptocurrency with completeness below 99%</li>
                        <li>Investigate price anomalies, especially if there are more than 10</li>
                        <li>Review OHLC inconsistencies as they may indicate data corruption</li>
                        <li>Check zero volume periods as they may represent exchange downtimes</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(html)
                
            logger.info(f"HTML report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def run_all_checks(self, symbol=None, generate_report=True, report_path="data_quality_report.html"):
        """
        Run all data quality checks on the specified symbol or all available symbols
        
        Args:
            symbol: Specific cryptocurrency symbol to check (e.g., 'BTC-USDC')
                   If None, checks all available cryptocurrencies
            generate_report: Whether to generate an HTML report
            report_path: Path to save the HTML report
            
        Returns:
            dict: Results of the data quality checks
        """
        # Reset results
        self.results = {}
        
        # Get all cryptocurrency directories
        crypto_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
                     
        if not crypto_dirs:
            logger.warning(f"No cryptocurrency directories found in {self.data_dir}")
            return {}
            
        # Filter for specific symbol if provided
        if symbol:
            crypto_symbol = symbol.replace('-', '_')
            crypto_dirs = [d for d in crypto_dirs if d == crypto_symbol]
            
            if not crypto_dirs:
                logger.error(f"No data directory found for {symbol}")
                return {}
        
        logger.info(f"Running data quality checks on {len(crypto_dirs)} cryptocurrencies")
        
        visualization_dir = "data_quality_reports/visualizations"
        Path(visualization_dir).mkdir(parents=True, exist_ok=True)
        
        # Process each cryptocurrency
        for crypto_dir in crypto_dirs:
            crypto_symbol = crypto_dir.replace('_', '-')
            master_file = f"{self.data_dir}/{crypto_dir}/master_data.csv"
            
            if not os.path.exists(master_file):
                logger.warning(f"No master data file found for {crypto_symbol}")
                continue
                
            logger.info(f"Processing {crypto_symbol}")
            
            # Read data
            try:
                df = pd.read_csv(master_file)
                
                # Run all checks
                self.check_data_continuity(df, crypto_symbol)
                self.detect_price_anomalies(df, crypto_symbol)
                self.verify_ohlc_consistency(df, crypto_symbol)
                self.analyze_volume_patterns(df, crypto_symbol)
                
                # Generate visualizations
                self.generate_visualizations(df, crypto_symbol, 
                                           output_dir=visualization_dir)
                                           
            except Exception as e:
                logger.error(f"Error processing {crypto_symbol}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate HTML report
        if generate_report and self.results:
            self.generate_html_report(output_path=report_path)
            
        return self.results
            
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Crypto Market Data Quality Check')
    
    parser.add_argument('--dir', type=str, default="market_data",
                      help='Directory containing cryptocurrency market data')
    parser.add_argument('--crypto', type=str,
                      help='Specific cryptocurrency to check (e.g., BTC-USDC)')
    parser.add_argument('--report', type=str, default="data_quality_report.html",
                      help='Path to save the HTML report')
    parser.add_argument('--no-report', action='store_true',
                      help='Skip generating HTML report')
                      
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    print(f"Crypto Market Data Quality Check")
    print(f"Checking data in: {args.dir}")
    if args.crypto:
        print(f"Focusing on: {args.crypto}")
    
    # Create checker and run checks
    checker = DataQualityChecker(data_dir=args.dir)
    results = checker.run_all_checks(
        symbol=args.crypto,
        generate_report=not args.no_report,
        report_path=args.report
    )
    
    # Print summary
    print("\nQuality Check Summary:")
    if results:
        for symbol, checks in results.items():
            print(f"\n{symbol}:")
            for check, value in checks.items():
                print(f"  {check}: {value}")
    else:
        print("No results to display. Check log for errors.")
        
    if not args.no_report and results:
        print(f"\nDetailed report saved to: {args.report}")
        print("Open this file in a browser to view the full report.")

if __name__ == "__main__":
    main()