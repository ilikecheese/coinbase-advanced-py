# GitHub Copilot Prompt: Cryptocurrency Range Trading Strategy

## Important Note for Copilot
I am not a programmer, so I need you to write ALL code completely and make it fully functional. For each step:
1. Write complete, runnable code (not just snippets or examples)
2. Run the code yourself to check for errors before showing it to me
3. Fix any errors you encounter
4. Provide clear instructions for me to run the code

For decision-making:
- Think through each technical decision carefully (consider multiple options)
- Choose the best implementation approach without asking me about coding decisions
- Only ask me about scope/feature decisions if truly necessary
- Be confident in your technical choices

For communication:
- Keep explanations brief and to the point
- Minimize word count in all replies
- Focus on practical information rather than theory
- Use bullet points when possible for clarity

## Overview
Create a Python application that implements and backtests a percentage-based range trading strategy for cryptocurrencies. The strategy buys when prices drop by X% and sells when prices rise by X%, taking advantage of volatility in sideways markets. The project should be built incrementally, with each component fully tested before moving to the next.

## Core Requirements

### Project Structure
- Implement using a modular architecture with clear separation of concerns
- Create a `DEVELOPMENT_PLAN.md` file listing implementation phases, components, and testing milestones
- Include specific checkpoints where functionality is verified before proceeding
- Design the system to allow for independent development and testing of components

### Data Management
- Read historical 1-minute candle data for USDC trading pairs from provided CSV files
- Initially focus on a single cryptocurrency pair for testing
- Include functionality to download additional historical data when needed

### Basic Strategy Implementation
1. Start with $100 in cash and $100 in cryptocurrency
2. When price increases by X% (configurable), sell X% of cryptocurrency holdings
3. When price decreases by X% (configurable), buy cryptocurrency worth X% of cash holdings
4. Track portfolio value (cash + crypto) over time
5. Include Coinbase Advanced API fee structure in calculations

### Risk Management
- Implement a minimum cash reserve threshold that pauses buying when reached
- Resume trading when market conditions become favorable again
- Track drawdowns and provide alerts for excessive losses

### Visualization (Incremental Additions)
Implement these visualizations one at a time, ensuring each works before adding the next:
1. Basic price chart with buy/sell transactions marked
2. Highlighted regions showing potential buy/sell zones based on threshold
3. Portfolio value over time chart
4. Comparison of optimal trades (with hindsight) vs. actual trades (using only past data)
5. Trade frequency heatmap
6. Drawdown visualization
7. Cash vs. cryptocurrency allocation stacked area chart
8. Threshold sensitivity analysis comparing different percentage values

## Implementation Phases

### Phase 1: Core Infrastructure
- Set up project structure with separate modules for data loading, strategy, backtesting, and visualization
- Implement basic data loading and preprocessing
- Create simple portfolio tracking mechanism
- Build basic backtest functionality without visualization
- Implement basic logging for debugging
- **Checkpoint**: Verify data loading and basic strategy execution with console output

### Phase 2: Basic Strategy and Simple Visualization
- Implement the core percentage-based buy/sell logic
- Add transaction fee calculation
- Create basic price chart with transaction markers
- Implement portfolio value tracking
- **Checkpoint**: Verify strategy executes correctly with at least one cryptocurrency

### Phase 3: Enhanced Analysis and Visualization
- Add buy/sell zone highlighting
- Implement optimal vs. probable trade comparison
- Add portfolio value chart
- **Checkpoint**: Verify visualization accuracy and usefulness

### Phase 4: Advanced Features
- Implement the remaining visualizations incrementally
- Add cryptocurrency selection capabilities
- Implement threshold sensitivity analysis
- Add risk management features
- **Checkpoint**: Test complete strategy with various parameters

### Phase 5: Optimization and Refinement
- Add functionality to find optimal parameters for different market conditions
- Implement performance reporting
- Create dashboard for easy strategy monitoring
- **Checkpoint**: Validate overall system performance and usability

## Technical Details
- Use Python 3.8+
- Required libraries: pandas, numpy, matplotlib/plotly for visualization
- Optional: dash for interactive dashboard
- Structure code using classes with clear responsibilities
- Include thorough error handling
- Add appropriate comments and documentation
- Provide installation instructions for all required packages
- Include a requirements.txt file
- Assume I will be running this on a standard setup (Windows/Mac) with minimal Python experience
- Write comprehensive error messages that clearly explain any problems
- Write README.md with clear instructions for setup and usage

## Example Usage
The application should be designed with parameters at the top of the main script file for easy configuration:

```python
# Configuration Parameters Section
PARAMETERS = {
    # Data settings
    "csv_file": "MATIC_USDC_1m.csv",  # Input data file
    "pair_name": "MATIC/USDC",        # Cryptocurrency pair name
    
    # Strategy parameters
    "threshold_pct": 5,               # Percentage threshold for buy/sell decisions
    "initial_cash": 100,              # Starting cash amount in USDC
    "initial_crypto": 100,            # Starting crypto amount (in USD value)
    "min_cash_reserve": 20,           # Minimum cash reserve (in USD)
    
    # Backtest settings
    "fee_pct": 0.1,                   # Trading fee percentage
    
    # Visualization options
    "show_buy_sell_zones": True,      # Highlight potential buy/sell zones
    "show_optimal_trades": True,      # Show comparison of optimal vs actual trades
    "show_portfolio_value": True,     # Display portfolio value chart
    "show_allocation_chart": True,    # Display cash vs crypto allocation chart
    
    # Optimization parameters (when running optimization mode)
    "run_optimization": False,        # Whether to run parameter optimization
    "threshold_range": [1, 2, 3, 4, 5, 7, 10],
    "min_cash_reserve_range": [10, 20, 30, 40, 50]
}

# The script should then be executable with: python range_trader.py
```

## Development Approach
Write complete, fully functional code since I am not a programmer. Start simple and build incrementally:

1. Write a complete working version of each component before moving to the next
2. Run the code yourself to check for errors, and fix any issues before showing me the code
3. Provide detailed comments throughout the code explaining what each section does
4. Include simple instructions for how to run each script
5. When adding new features, ensure they don't break existing functionality
6. If you encounter errors or issues while running the code, fix them before proceeding

After implementing each component, test thoroughly before moving on. If complexity becomes overwhelming, refer to the DEVELOPMENT_PLAN.md file to identify independent components that can be refactored or simplified.

## Error Prevention
- Implement checkpoints throughout the code to validate intermediate results
- Use assertions to catch logical errors early
- Create visualization tools to help debug strategy behavior
- Log all transactions and state changes for audit purposes
- Add resume capability to allow stopping and continuing development