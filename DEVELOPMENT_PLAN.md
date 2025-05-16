# Development Plan for Cryptocurrency Range Trading Strategy

## Phase 1: Core Infrastructure
- [x] Project structure and module scaffolding
- [x] Data loader for 1-minute USDC pair CSVs
- [x] Simple portfolio tracker
- [x] Basic backtest runner (no visualization)
- [x] Logging and error handling
- [x] Checkpoint: Console output for data loading and basic strategy
  - Console output and logging verified
  - Only the 'price with buy/sell trades' plot is used for visualization (all others removed)

## Phase 2: Basic Strategy and Visualization
- [x] 2. Implement buy/sell logic
- [x] 1. Add transaction fee calculation
    - Fee structure (Coinbase Advanced):
        - $0-$1K monthly: Maker 0.60%, Taker 1.20%
        - $1K-$500K monthly: Maker 0.35%, Taker 0.75%
    - Code uses these values for fee calculations.
- [x] 3. Basic price chart with transaction markers
- [x] 4. Portfolio value tracking
    - Trade output now includes: USD value, qty, fee, crypto balance, USDC balance, and portfolio value columns for each trade.
- [ ] 5. Checkpoint: Strategy executes with one crypto

## Phase 3: Enhanced Analysis and Visualization
- [ ] Buy/sell zone highlighting
- [ ] Optimal vs. probable trade comparison
- [ ] Portfolio value chart
- [ ] Checkpoint: Visualization accuracy

## Phase 4: Advanced Features
- [ ] Remaining visualizations
- [ ] Crypto selection
- [ ] Threshold sensitivity analysis
- [ ] Risk management
- [ ] Checkpoint: Test with various parameters

## Phase 5: Optimization and Refinement
- [ ] Parameter optimization
- [ ] Performance reporting
- [ ] Dashboard
- [ ] Checkpoint: Validate system performance
