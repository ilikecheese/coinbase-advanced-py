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
- [x] Implement buy/sell logic
- [x] Add transaction fee calculation (Coinbase Advanced fee structure)
- [x] Basic price chart with transaction markers
- [x] Portfolio value tracking (trade output includes USD value, qty, fee, crypto balance, USDC balance, and portfolio value columns for each trade)
- [x] Checkpoint: Strategy executes with one crypto

## Phase 3: Risk Management (Stop-Loss & Take-Profit)
- [x] Implement stop-loss and take-profit logic:
    - For each trade, set stop_loss_price and take_profit_price based on entry price and strategy params.
    - Monitor each candle for stop-loss/take-profit triggers before processing new trades.
    - Extend trade log to include entry_price, stop_loss_price, take_profit_price, and exit_reason.
    - Update portfolio and trade history upon exit, applying fees.
- [x] Add stop_loss_pct and take_profit_pct to strategy_params (set to None to disable).
- [x] Visualize stop-loss/take-profit exits on price chart (distinct markers).
- [x] Test with known price movements and real data to verify triggers and fee application.
- [x] Checkpoint: Risk management logic integrated and validated.

## Phase 4: Dynamic Thresholds with ATR
- [ ] Replace static threshold_pct with dynamic ATR-based thresholds:
    - Calculate ATR (Average True Range) and convert to percentage of price.
    - Set buy/sell thresholds as k * ATR% (scaling factor k configurable).
    - Add buy_threshold and sell_threshold columns to OHLCV DataFrame.
    - Update strategy logic to use dynamic thresholds for trade triggers.
    - Add atr_period and atr_k to strategy_params.
- [ ] Validate ATR calculations against a known library (e.g., ta, pandas_ta).
- [ ] Test backtests with different atr_period and atr_k values.
- [ ] Checkpoint: Dynamic thresholds adapt to volatility and pass tests.

## Phase 5: Portfolio Value Chart
- [ ] Compute portfolio value (cash + crypto * close_price) for each candle.
- [ ] Store portfolio value in DataFrame or separate array.
- [ ] Create dual-axis matplotlib plot: portfolio value (line), price (with trade markers).
- [ ] Add gridlines, labels, legend; save plot to plots directory.
- [ ] Verify calculations and plot rendering.
- [ ] Checkpoint: Portfolio value chart accurate and informative.

## Phase 6: Grid Search for Parameter Optimization
- [ ] Define parameter grid (e.g., threshold_pct, trade_size_pct, atr_k, etc.).
- [ ] Use itertools.product to generate parameter combinations.
- [ ] Run backtest for each combination; compute metrics (total return, Sharpe ratio, max drawdown).
- [ ] Save results to CSV (grid_search_results.csv) and optionally plot heatmaps.
- [ ] Test with small grid to verify correctness.
- [ ] Checkpoint: Grid search framework operational and results validated.

## Phase 7: Advanced Analysis and Visualization
- [ ] Buy/sell zone highlighting (visualize threshold-based buy/sell regions on price chart)
- [x] Optimal vs. probable trade comparison (now strategy-constrained, not just local min/max)
- [ ] Remaining visualizations (allocation chart, drawdown, heatmap, etc.)
- [ ] Crypto selection (allow user to select or batch test multiple cryptos)
- [ ] Threshold sensitivity analysis (compare results for different thresholds)
- [ ] Risk management (drawdown tracking, alerts, etc.)
- [ ] Checkpoint: Test with various parameters and visualizations

## Phase 8: Optimization and Refinement
- [ ] Parameter optimization (automated search for best parameters)
- [ ] Performance reporting (summary stats, trade stats, etc.)
- [ ] Dashboard (integrated reporting/visualization)
- [ ] Checkpoint: Validate system performance

---

**Technical Notes:**
- New modules to be created: `atr_threshold.py` (ATR/dynamic thresholds), `risk_management.py` (stop-loss/take-profit), `grid_search.py` (parameter optimization).
- Integration and testing required at each phase.
- Prioritize: 1) Stop-Loss/Take-Profit, 2) Dynamic ATR Thresholds, 3) Portfolio Value Chart, 4) Grid Search.
- See technical descriptions in project notes for implementation details.
