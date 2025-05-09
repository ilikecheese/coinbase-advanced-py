#!/usr/bin/env python3
"""
Test Candle Granularity

A simplified script to test why we're getting 6-hour candles when requesting 5-minute granularity.
"""

import time
from datetime import datetime, timedelta
import pandas as pd
import traceback
from coinbase.rest import RESTClient
from config import API_KEY, API_SECRET

def test_granularity(symbol="BTC-USDC", granularity="FIVE_MINUTE", hours=12):
    """Test a specific granularity setting"""
    print(f"\nTesting {granularity} candles for {symbol} over past {hours} hours...")
    
    try:
        client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        end_timestamp = int(end_time.timestamp())
        start_timestamp = int(start_time.timestamp())
        
        print(f"Time range: {start_time} to {end_time}")
        print(f"Timestamps: {start_timestamp} to {end_timestamp}")
        
        # Make API request
        print("Requesting candles from Coinbase API...")
        candles_response = client.get_public_candles(
            product_id=symbol,
            start=str(start_timestamp),
            end=str(end_timestamp),
            granularity=granularity
        )
        
        # Check if we got candles
        if not hasattr(candles_response, 'candles') or not candles_response.candles:
            print(f"ERROR: No candle data available for {symbol}")
            return
            
        print(f"SUCCESS: Received {len(candles_response.candles)} candles from API")
        
        # Print the first few candles to see what we're getting
        print("\nFirst 5 candles:")
        for i, candle in enumerate(candles_response.candles[:5]):
            dt = datetime.fromtimestamp(int(candle.start))
            print(f"  {i+1}. {dt} - Open: {candle.open}, Close: {candle.close}")
            
        # Analyze time differences
        if len(candles_response.candles) > 1:
            print("\nTime differences between consecutive candles:")
            time_diffs = []
            
            for i in range(1, min(5, len(candles_response.candles))):
                time_diff = int(candles_response.candles[i].start) - int(candles_response.candles[i-1].start)
                minutes_diff = time_diff / 60
                
                # Convert timestamps to readable dates
                dt1 = datetime.fromtimestamp(int(candles_response.candles[i-1].start))
                dt2 = datetime.fromtimestamp(int(candles_response.candles[i].start))
                
                print(f"  Candles {i} to {i+1}: {minutes_diff:.0f} minutes ({dt1} to {dt2})")
                time_diffs.append(time_diff)
            
            # Calculate average time between candles
            if time_diffs:
                avg_diff_seconds = sum(time_diffs) / len(time_diffs)
                avg_diff_minutes = avg_diff_seconds / 60
                avg_diff_hours = avg_diff_minutes / 60
                
                print(f"\nAVERAGE TIME BETWEEN CANDLES:")
                print(f"  {avg_diff_seconds:.0f} seconds")
                print(f"  {avg_diff_minutes:.1f} minutes")
                print(f"  {avg_diff_hours:.2f} hours")
                
                # Compare with expected times
                granularity_minutes = {
                    "ONE_MINUTE": 1,
                    "FIVE_MINUTE": 5,
                    "FIFTEEN_MINUTE": 15,
                    "THIRTY_MINUTE": 30,
                    "ONE_HOUR": 60,
                    "TWO_HOUR": 120,
                    "SIX_HOUR": 360,
                    "ONE_DAY": 1440
                }
                
                expected = granularity_minutes.get(granularity, 0)
                print(f"\nExpected minutes between candles for {granularity}: {expected}")
                
                if abs(avg_diff_minutes - expected) < 1:
                    print(f"RESULT: ✅ Granularity is correct ({granularity})")
                else:
                    # Try to determine what granularity we actually got
                    for gran_name, gran_mins in granularity_minutes.items():
                        if abs(avg_diff_minutes - gran_mins) < gran_mins * 0.1:  # Within 10% of expected
                            print(f"RESULT: ❌ Requested {granularity} but got {gran_name} candles")
                            break
                    else:
                        print(f"RESULT: ❌ Requested {granularity} but got unknown granularity ({avg_diff_minutes:.1f} minutes)")

            # Save to CSV
            data = []
            for candle in candles_response.candles:
                data.append({
                    'timestamp': int(candle.start),
                    'datetime': datetime.fromtimestamp(int(candle.start)),
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': float(candle.volume)
                })
                
            df = pd.DataFrame(data)
            output_file = f"{symbol.replace('-', '_')}_{granularity.lower()}_test.csv"
            df.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
            
    except Exception as e:
        print(f"ERROR testing {granularity} for {symbol}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting candle granularity tests...")
    
    # Test different combinations
    try:
        test_granularity("BTC-USDC", "FIVE_MINUTE", 2)  # 2 hours of BTC-USDC with 5-min candles
        test_granularity("ETH-USDC", "FIVE_MINUTE", 2)  # 2 hours of ETH-USDC with 5-min candles
        test_granularity("BTC-USDC", "ONE_MINUTE", 1)   # 1 hour of BTC-USDC with 1-min candles
    except Exception as e:
        print(f"Error in main test sequence: {e}")
        traceback.print_exc()
    
    print("\nCandle granularity tests completed")