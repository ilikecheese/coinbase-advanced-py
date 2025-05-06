#!/usr/bin/env python3
"""
List all Coinbase accounts with simplified output:
- Currency
- Currency Balance
- USD Equivalent
"""

from coinbase.rest import RESTClient
from config import API_KEY, API_SECRET

def format_usd_value(value):
    """Format USD value to show zeros until there's a significant digit"""
    if value == 0:
        return "$0.00"
    
    if value >= 0.01:
        return f"${value:.2f}"
    
    # For very small values, find the first significant digit
    decimal_str = f"{value:.20f}"
    # Find the first non-zero digit after decimal
    position = 2  # Start after "0."
    while position < len(decimal_str) and decimal_str[position] == '0':
        position += 1
    
    if position >= len(decimal_str):
        return "$0.00"  # No significant digits found within precision
        
    # Show zeros until the first significant digit plus one more
    precision = position - 1  # -1 because decimal point is counted in position
    
    if precision > 10:
        # For extremely small values, use scientific notation
        return f"${value:.2e}"
    else:
        return f"${value:.{precision + 2}f}"  # +2 to show two significant digits

def main():
    # Initialize the REST client with credentials
    client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
    
    try:
        print("Fetching accounts from Coinbase Advanced API...")
        
        # Initialize variables for pagination
        cursor = None
        all_accounts = []
        page_count = 0
        limit = 250  # Maximum allowed by API
        
        # Loop through all pages of accounts
        while True:
            page_count += 1
            
            # Get accounts with pagination parameters
            if cursor:
                accounts = client.get_accounts(cursor=cursor, limit=limit)
            else:
                accounts = client.get_accounts(limit=limit)
            
            # Check if accounts attribute exists
            if not hasattr(accounts, 'accounts') or not accounts.accounts:
                break
                
            # Add accounts from this page to our collection
            all_accounts.extend(accounts.accounts)
            
            # Check if there are more pages
            if hasattr(accounts, 'has_next') and accounts.has_next and hasattr(accounts, 'cursor'):
                cursor = accounts.cursor
            else:
                break
        
        # Get cryptocurrency prices
        print("Fetching current market prices...")
        prices = {}
        
        # Get all tradable products first
        products = client.get_products()
        
        # Create a price dictionary for quick lookup
        for product in products.products:
            if product.product_id.endswith('-USD'):
                base_currency = product.product_id.split('-')[0]
                if hasattr(product, 'price') and product.price:
                    prices[base_currency] = float(product.price)
        
        # Add USD price as 1.0
        prices['USD'] = 1.0
        prices['USDC'] = 1.0  # Assuming USDC is pegged to USD
        prices['USDT'] = 1.0  # Assuming USDT is pegged to USD
        
        # Sort accounts by USD value (highest to lowest)
        def get_usd_value(account):
            """Calculate the USD value of an account"""
            currency = account.currency if hasattr(account, 'currency') else 'N/A'
            balance = 0.0
            try:
                if hasattr(account, 'available_balance') and account.available_balance:
                    balance = float(account.available_balance.get('value', '0'))
            except (AttributeError, TypeError, ValueError):
                balance = 0.0
                
            price = prices.get(currency, 0.0)
            return balance * price

        # Explicitly sort by USD value, highest first        
        print("Sorting accounts by USD value (highest first)...")
        all_accounts.sort(key=get_usd_value, reverse=True)
        
        # Count non-zero balances
        non_zero_accounts = sum(1 for acc in all_accounts if get_usd_value(acc) > 0)
        
        # Print simplified account information
        print(f"\n{'=' * 75}")
        print(f"{'CURRENCY':<10} {'BALANCE':<25} {'USD EQUIVALENT':<20}")
        print(f"{'-' * 75}")
        
        total_usd_value = 0.0
        account_count = 0
        
        for account in all_accounts:
            # Get the currency
            currency = account.currency if hasattr(account, 'currency') else 'N/A'
            
            # Get the balance value
            balance = 0.0
            balance_str = '0'
            try:
                if hasattr(account, 'available_balance') and account.available_balance:
                    balance_str = account.available_balance.get('value', '0')
                    balance = float(balance_str)
            except (AttributeError, TypeError, ValueError):
                balance = 0.0
                balance_str = '0'
            
            # Calculate USD equivalent
            price = prices.get(currency, 0.0)
            usd_value = balance * price
            total_usd_value += usd_value
            
            # Skip zero balance accounts
            if balance == 0:
                continue
                
            account_count += 1
            
            # Format the output with special formatting for USD value
            formatted_usd = format_usd_value(usd_value)
            print(f"{currency:<10} {balance_str:<25} {formatted_usd:<20}")
        
        print(f"{'-' * 75}")
        print(f"{'TOTAL USD:':<10} {'':<25} ${total_usd_value:.2f}")
        print(f"{'=' * 75}")
        print(f"Showing {account_count} accounts with non-zero balances (out of {len(all_accounts)} total accounts)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()