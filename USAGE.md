# GridStrategyV4 — Complete Usage Guide

Every way to run, configure, optimize, and extend the V4.2 grid trading engine.

---

## Table of Contents

- [Installation](#installation)
- [Live Trading (live_trade.py)](#live-trading-live_tradepy)
- [Running Backtests (main.py)](#running-backtests-mainpy)
- [Running the Optimizer (optimizer.py)](#running-the-optimizer-optimizerpy)
- [Applying Optimized Parameters (apply_optimized_params.py)](#applying-optimized-parameters-apply_optimized_paramspy)
- [Using BacktestSimulator Programmatically](#using-backtestsimulator-programmatically)
- [Using GridStrategyV4 Directly in Code](#using-gridstrategyv4-directly-in-code)
- [Running Tests](#running-tests)
- [Configuration Reference](#configuration-reference)
- [Data Management](#data-management)
- [Output Files](#output-files)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

```bash
cd /path/to/GRID_Trade

# (Optional) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `ccxt` | Exchange data fetching + live order placement |
| `pandas` | DataFrame operations |
| `numpy` | Numerical computation |
| `optuna` | Bayesian parameter optimization |
| `matplotlib` | Chart generation |
| `pyarrow` | Parquet file caching |
| `pytest` | Test runner |
| `python-dotenv` | Environment variable management (live trading) |

---

## Live Trading (live_trade.py)

Trade live on Binance USDM Perpetual Futures using the same strategy logic as the backtest engine.

### Environment Setup

**1. API Keys**

Set your Binance API credentials as environment variables:

```bash
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_api_secret_here"
```

For persistent setup, add these to your shell profile (`~/.zshrc`, `~/.bashrc`) or use a `.env` file in your project directory.

**Requirements**: Your Binance account must have Futures trading enabled and be set to **Hedge Mode** (the bot will attempt to enable this automatically).

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--coin` | string | *required* | Coin ticker (BTC, ETH, SOL, etc.) |
| `--capital` | float | Full wallet | Capital allocation in USDT |
| `--leverage` | int | From config | Override leverage multiplier |
| `--dry-run` | flag | off | Simulate orders without sending to exchange |
| `--resume` | flag | off | Restore from saved state after crash/restart |
| `--testnet` | flag | off | Use Binance Futures testnet |
| `--no-short` | flag | off | Disable short grid (long-only) |
| `--log-level` | string | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Quick Start

```bash
# Dry-run first (simulates orders, reads real market data)
python3 live_trade.py --coin BTC --dry-run

# Testnet (real orders on Binance testnet — no real money)
python3 live_trade.py --coin BTC --testnet --capital 1000

# Live trading with custom capital and leverage
python3 live_trade.py --coin BTC --capital 5000 --leverage 10

# Long-only mode
python3 live_trade.py --coin ETH --capital 3000 --no-short

# Debug logging for troubleshooting
python3 live_trade.py --coin SOL --dry-run --log-level DEBUG
```

### How It Works

The live runner replicates the backtest's 11-step strategy loop on real 15-minute candles:

1. Maintains a rolling buffer of 200 candles for indicator computation
2. Waits for each 15m candle close, then runs the full strategy step
3. Detects order fills by polling the exchange (orders tracked → missing = filled)
4. Places grid orders as real limit orders on Binance
5. Executes stops, prunes, and emergency closes as market orders

**Strategy steps** (identical to backtest):
Regime detection → Circuit breaker → Stop loss → Liquidation check → Pruning → Trailing up → VaR check → Grid generation → Fill sync → Funding rate → Equity logging

### Dry-Run Mode

Dry-run mode is the recommended first step. It:
- Reads real market data (candles, ticker, funding rates, positions, balance)
- Simulates all order placements (logs them, returns synthetic order IDs)
- Runs the full strategy loop with real indicators
- Logs all decisions and trade events

This lets you verify the strategy is working correctly before committing real capital.

### State Persistence and Crash Recovery

All state is saved to `data/live_state/` after every strategy iteration:

| File | Contents |
|------|----------|
| `{SYMBOL}_state.json` | Full runner state (positions, grid anchors, metrics, active orders) |
| `{SYMBOL}_trades.csv` | Trade log with columns: timestamp, price, qty, label, pnl, fees, etc. |
| `{SYMBOL}_trades.jsonl` | Same trades in JSON lines format (machine-readable) |
| `{SYMBOL}_snapshot_*.json` | Timestamped checkpoints (last 10 kept) |

**To recover after a crash or restart:**

```bash
python3 live_trade.py --coin BTC --resume
```

This restores positions, grid state, metrics, and the candle buffer from the last saved state. The bot re-syncs with the exchange to pick up any fills that occurred while it was offline.

### Monitoring and Logs

**Log files** are written to `data/live_logs/{SYMBOL}.log` with rotation (10MB, 5 backups).

**Console output** shows formatted trade events:

```
[2024-01-15 14:30:00] SELL_CLOSE_LONG | BTC/USDT
  Price: $43,250.00 | Qty: 0.023 | PnL: +$12.45 (+0.57%)
  Hold: 2.5h | R: +1.23 | Regime: NOISE

[2024-01-15 14:45:00] EQUITY | $10,234.56 | Wallet: $10,100.00
  UPnL_L: +$80.00 | UPnL_S: +$54.56 | Regime: NOISE
```

**Health monitoring** runs every iteration:
- Heartbeat tracking (alerts if >20 min gap between iterations)
- Position sync verification (internal tracker vs exchange positions)
- Equity monitoring with warning (10% drawdown) and critical (25% drawdown) thresholds
- Consecutive error tracking (>5 consecutive errors triggers shutdown)

### Safety Features

| Feature | Description |
|---------|-------------|
| **Graceful shutdown** | SIGINT/SIGTERM cancels open orders, saves state, keeps positions |
| **Emergency shutdown** | On critical equity loss, market-closes ALL positions |
| **Idempotent shutdown** | Signal handler + KeyboardInterrupt won't double-execute |
| **Position reconciliation** | Exchange positions are the source of truth; internal state adjusts |
| **Atomic state writes** | State written to `.tmp` then renamed — no corruption on crash |
| **Order verification** | Disappeared orders verified via `fetch_order()` before assuming fill |
| **Min notional check** | Orders below exchange minimum are rejected before sending |
| **Retry with backoff** | Network errors and rate limits retry with exponential backoff |

### Stopping the Bot

- **Ctrl+C** (SIGINT): Cancels all open orders, saves state, keeps positions open. You can resume later with `--resume`.
- **Kill process** (SIGTERM): Same graceful shutdown behavior.
- Positions remain open on the exchange after shutdown. Manage them manually via Binance UI if needed.

---

## Running Backtests (main.py)

The primary way to run the strategy.

### Basic Usage

```bash
# Default: BTC, ETH, SOL from 2024-01-01 with $10,000 capital
python3 main.py
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--coins` | list | `BTC ETH SOL` | Space-separated coin tickers |
| `--start` | string | `2024-01-01` | Start date (YYYY-MM-DD) |
| `--end` | string | Current date | End date (YYYY-MM-DD) |
| `--capital` | float | `10000` | Initial capital in USD |
| `--no-plot` | flag | off | Skip chart generation |
| `--no-short` | flag | off | Disable short grid (long-only mode) |

### Examples

```bash
# Single coin, custom date range
python3 main.py --coins BTC --start 2023-01-01 --end 2024-12-31

# Multiple coins with custom capital
python3 main.py --coins BTC ETH SOL DOGE --start 2024-01-01 --capital 50000

# Long-only mode (no short grid)
python3 main.py --coins ETH --no-short --start 2024-06-01

# Fast run without charts
python3 main.py --coins BTC --no-plot

# Full-form coin symbol (also accepted)
python3 main.py --coins BTC/USDT ETH/USDT

# Combine all options
python3 main.py --coins SOL --start 2024-03-01 --end 2024-09-01 --capital 5000 --no-short --no-plot
```

### What It Does

For each coin:
1. Fetches OHLCV data from Binance (or loads from Parquet cache)
2. Runs GridStrategyV4 with parameters from `config.py`
3. Prints a formatted performance summary to console
4. Exports trade log to `data/output/{COIN}_trades.csv`
5. Generates equity curve + trade map charts to `data/charts/`

### Console Output

```
GridStrategyV4 | Hedge Mode | KAMA Regime | 15m Candles
Capital: $10,000 | Leverage: 1x | Short: ON
Coins: ['BTC'] | Period: 2024-01-01 → now

Fetching BTC/USDT...
  35040 candles loaded (1704067200000 → 1739491200000)

────────────────────────────────────────────────────────────
  BTC/USDT — Performance Summary
────────────────────────────────────────────────────────────
  Return             +12.45%
  Buy & Hold         +85.30%
  Max Drawdown        8.23%
  Sharpe              1.234
  ...
```

---

## Running the Optimizer (optimizer.py)

Finds optimal parameters using anchored walk-forward validation with Optuna, then selects a **robust parameter cluster** (top-10-percentile median) instead of the single best trial to reduce overfitting.

### Basic Usage

```bash
python3 optimizer.py --coins BTC SOL --trials 200
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--coins` | list | `BTC/USDT ETH/USDT SOL/USDT` | Coins to optimize across |
| `--trials` | int | `100` | Number of Optuna trials |
| `--start` | string | `2024-01-01` | Data start date |
| `--end` | string | Current date | Data end date |
| `--windows` | int | `4` | Walk-forward windows |
| `--study-name` | string | `grid_v4_wfo` | Optuna study name (for resume) |
| `--mc-perms` | int | `50` | Monte Carlo permutations |
| `--skip-mc` | flag | off | Skip Monte Carlo validation |

### Examples

```bash
# Quick optimization (50 trials, skip MC for speed)
python3 optimizer.py --coins BTC --trials 50 --skip-mc

# Full optimization across 3 coins
python3 optimizer.py --coins BTC ETH SOL --trials 500 --windows 4

# Custom date range
python3 optimizer.py --coins SOL --start 2023-06-01 --end 2024-06-01 --trials 200

# Resume a previous study (same study name)
python3 optimizer.py --coins BTC SOL --trials 100 --study-name my_study
python3 optimizer.py --coins BTC SOL --trials 100 --study-name my_study  # adds 100 more
```

### What It Does

1. Loads OHLCV data for all specified coins
2. For each Optuna trial:
   - Samples parameters from `OPTIMIZER_SPACE` in `config.py`
   - Splits data into anchored walk-forward windows (70% train / 30% test)
   - Evaluates on **train** data (TPE learns from this)
   - Stores **test** scores as held-out attributes (no leakage)
   - Scores using pessimistic (min) Calmar × sqrt(PF) across all windows
3. After all trials:
   - **Robust parameter selection**: takes the top-10% of trials, computes the **median** of each parameter across the cluster
   - Prints a parameter stability report (Coefficient of Variation per param)
   - Runs OOS validation and Monte Carlo significance test on the **robust params**
4. Saves results to `data/optimizer_results_robust.json` (primary output)
5. Also saves single-best trial to `data/best_params_v4.json` (reference only)

### Robust Parameter Report (Sample Output)

```
===== ROBUST PARAMETER REPORT =====
(Top 10% of 200 trials = 20 trials used)

Robust Parameters (median of top cluster):
  grid_spacing_k             0.4200  (default: 0.4)
  gamma                      0.9800  (default: 1.2)
  ...

Parameter Stability (CV — lower is more stable):
  grid_spacing_k        CV=0.1823  ████              [STABLE]
  gamma                 CV=0.4950  ██████████        [MODERATE]
  ...

Overall Robustness Score: 0.3012
Interpretation: GOOD robustness — parameters generalise well
```

Interpretation of the robustness score:

| Score | Interpretation |
|-------|----------------|
| `< 0.25` | **EXCELLENT** — parameters are very stable across top trials |
| `< 0.40` | **GOOD** — parameters generalise well |
| `≤ 0.50` | **ACCEPTABLE** — monitor live performance |
| `> 0.50` | **WARNING** — likely overfitting, widen data or reduce params |

### Optimizable Parameters

Defined in `OPTIMIZER_SPACE` in `config.py`:

| Parameter | Type | Range | Category |
|-----------|------|-------|----------|
| `grid_spacing_k` | float | 0.8 – 2.5 | Grid Structure |
| `spacing_floor` | float | 0.004 – 0.010 | Grid Structure |
| `grid_levels` | int | 2 – 6 | Grid Structure |
| `regen_drift_mult` | float | 1.0 – 4.0 | Grid Structure |
| `adaptive_floor_scale` | float | 0.5 – 3.0 | Grid Structure |
| `tp_concentration` | float | 0.4 – 0.9 | Grid Structure |
| `gamma` | float | 0.3 – 2.0 | Inventory Control |
| `kappa` | float | 0.5 – 3.0 | Inventory Control |
| `max_inventory_per_side` | int | 2 – 6 | Inventory Control |
| `as_time_horizon` | float | 32 – 288 | Inventory Control |
| `regime_hysteresis_bars` | int | 1 – 6 | Regime Detection |
| `adx_trend_threshold` | float | 20.0 – 35.0 | Regime Detection (**new V4.2**) |
| `atr_sl_mult` | float | 2.0 – 5.0 | Risk Management |
| `max_position_pct` | float | 0.3 – 0.7 | Risk Management |
| `stop_cooldown_bars` | int | 24 – 96 | Risk Management (**new V4.2**) |
| `low_volume_threshold` | float | 0.30 – 0.70 | Risk Management (**new V4.2**) |
| `deviance_sigma` | float | 2.5 – 6.0 | Pruning |
| `gap_prune_mult` | float | 2.5 – 7.0 | Pruning |
| `offset_prune_ratio` | float | 0.8 – 4.0 | Pruning |
| `prune_cooldown_bars` | int | 1 – 8 | Pruning |
| `kelly_window` | int | 30 – 100 | Sizing (**new V4.2**) |
| `funding_harvest_threshold` | float | 0.0001 – 0.0005 | Funding (**new V4.2**) |

**Hardcoded (not optimized)** — these are domain/exchange constants, not strategy levers:
- `adx_period = 14` (Wilder's canonical), `kelly_fraction = 0.25` (Quarter-Kelly theory), `round_number_nudge` (market microstructure), `fee_maker/fee_taker` (exchange fees), `VaR_deleverage_ratio = 0.75` (risk boundary), `high_corr_threshold = 0.80` (empirical crisis correlation)

Commented-out parameters in `OPTIMIZER_SPACE` are fixed at their `STRATEGY_PARAMS` values. Uncomment to include them in search.

### Persistence

Optuna stores all trials in `grid_v4_wfo.db` (SQLite). You can:
- Resume a study by using the same `--study-name`
- Run multiple sessions that accumulate knowledge
- Delete `grid_v4_wfo.db` to start fresh

---

## Applying Optimized Parameters (apply_optimized_params.py)

After optimization, apply the robust parameters to `config.py`.

### Preview Changes (Dry Run)

```bash
python3 apply_optimized_params.py --preview
```

Shows what would change without modifying any file.

### Apply Robust Parameters (Recommended)

```bash
# Apply from the robust JSON output (recommended)
python3 apply_optimized_params.py --params data/optimizer_results_robust.json --key robust_params
```

### Apply Single-Best Parameters (Reference Only)

```bash
python3 apply_optimized_params.py --params data/best_params_v4.json
```

### Auto-Confirm

```bash
python3 apply_optimized_params.py --params data/optimizer_results_robust.json --key robust_params --yes
```

Skips the confirmation prompt.

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--params` | string | `data/best_params_v4.json` | Path to parameters JSON |
| `--key` | string | `best_params` | Key inside JSON to read params from |
| `--config` | string | `config.py` | Path to config file to update |
| `--preview` | flag | off | Dry run (show changes only) |
| `--yes` / `-y` | flag | off | Auto-confirm changes |

### Custom Parameters File

```bash
# Apply from a specific file
python3 apply_optimized_params.py --params data/my_custom_params.json

# Apply to a different config
python3 apply_optimized_params.py --config config_staging.py --preview
```

---

## Using BacktestSimulator Programmatically

**File**: `backtest/simulator.py`

For scripting and notebooks.

### Single Coin

```python
from backtest.simulator import BacktestSimulator
from config import STRATEGY_PARAMS

# Use default config
sim = BacktestSimulator()
result = sim.run_single('BTC/USDT', start='2024-01-01')

print(result['metrics']['total_return_pct'])
print(result['metrics']['sharpe_ratio'])
print(result['metrics']['max_drawdown_pct'])
```

### Multiple Coins

```python
sim = BacktestSimulator()
all_results = sim.run_multi(
    coins=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    start='2024-01-01',
    end='2024-12-31'
)

# Per-coin results
for coin, result in all_results.items():
    if coin == '_portfolio':
        continue
    print(f"{coin}: {result['metrics']['total_return_pct']:+.2f}%")

# Aggregated portfolio metrics
portfolio = all_results['_portfolio']
print(f"Portfolio Return: {portfolio['metrics']['total_return_pct']:+.2f}%")
```

### Custom Configuration

```python
from config import STRATEGY_PARAMS

custom_config = STRATEGY_PARAMS.copy()
custom_config.update({
    'initial_capital': 50000,
    'grid_levels': 15,
    'gamma': 0.3,
    'allow_short': False,
    'trailing_enabled': True,
})

sim = BacktestSimulator(config=custom_config)
result = sim.run_single('ETH/USDT', start='2024-01-01')
```

### Custom Backtest Config

```python
custom_bt_config = {
    'coins': ['BTC/USDT', 'SOL/USDT'],
    'start_date': '2023-06-01',
    'end_date': '2024-06-01',
    'timeframe': '15m',
}

sim = BacktestSimulator(
    config=STRATEGY_PARAMS,
    backtest_config=custom_bt_config
)
results = sim.run_multi()
```

---

## Using GridStrategyV4 Directly in Code

**File**: `engine/strategy.py`

For maximum control.

### Basic Usage

```python
import pandas as pd
from engine.strategy import GridStrategyV4
from config import STRATEGY_PARAMS
from backtest.data_fetcher import fetch_data

# Fetch data
df = fetch_data('BTC/USDT', '15m', '2024-01-01')

# Create strategy
config = STRATEGY_PARAMS.copy()
config['initial_capital'] = 10000
strat = GridStrategyV4(config)

# Run backtest
result = strat.run(df)

# Access results
equity_curve = result['equity_curve']    # numpy array, length = len(df)
trades = result['trades']                # list of trade dicts
regime_log = result['regime_log']        # numpy array of regime codes per bar
metrics = result['metrics']              # dict of performance metrics
```

### Accessing Trade Data

```python
import pandas as pd

trades_df = pd.DataFrame(result['trades'])
trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='ms')

# Filter by type
longs_opened = trades_df[trades_df['label'] == 'BUY_OPEN_LONG']
stops = trades_df[trades_df['label'].str.contains('STOP')]
prunes = trades_df[trades_df['label'].str.contains('PRUNE')]

# Profitable trades only
profitable = trades_df[trades_df['pnl'] > 0]
print(f"Average win: ${profitable['pnl'].mean():.2f}")
print(f"Largest win: ${profitable['pnl'].max():.2f}")
```

### Accessing Equity Curve

```python
import numpy as np

ec = result['equity_curve']

# Max drawdown
peak = np.maximum.accumulate(ec)
drawdown = (peak - ec) / peak * 100
print(f"Max Drawdown: {drawdown.max():.2f}%")

# Returns
returns = np.diff(ec) / ec[:-1]
print(f"Daily return (avg): {np.mean(returns) * 96:.4f}%")  # 96 bars per day
```

### Regime Analysis

```python
from core.kama import REGIME_NAMES

regime_log = result['regime_log']

# Count bars per regime
for code, name in REGIME_NAMES.items():
    count = np.sum(regime_log == code)
    pct = count / len(regime_log) * 100
    print(f"{name}: {count} bars ({pct:.1f}%)")
```

### Parameter Sweep (Manual)

```python
from engine.strategy import GridStrategyV4
from backtest.data_fetcher import fetch_data
from config import STRATEGY_PARAMS

df = fetch_data('BTC/USDT', '15m', '2024-01-01')

results = {}
for gamma in [0.1, 0.3, 0.5, 1.0, 2.0]:
    config = STRATEGY_PARAMS.copy()
    config['gamma'] = gamma
    strat = GridStrategyV4(config)
    result = strat.run(df)
    results[gamma] = result['metrics']
    print(f"gamma={gamma}: Return={result['metrics']['total_return_pct']:+.2f}%, "
          f"DD={result['metrics']['max_drawdown_pct']:.2f}%")
```

### Using Individual Modules

```python
# Indicators only
from core.kama import calculate_er, calculate_kama, detect_regime
from core.atr import calculate_atr, calculate_z_score, calculate_rolling_volatility

import numpy as np
closes = df['close'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)

er = calculate_er(closes, period=10)
kama = calculate_kama(closes, er, fast_period=2, slow_period=30)
atr = calculate_atr(highs, lows, closes, period=14)
z = calculate_z_score(closes, atr, period=20)
vol = calculate_rolling_volatility(closes, period=14)
regime = detect_regime(kama, er, atr, threshold=0.15, er_trend_thresh=0.5)
```

```python
# Grid generation only
from core.grid import generate_grid_levels, calculate_dynamic_spacing

spacing = calculate_dynamic_spacing(atr=500, spacing_k=1.0, spacing_floor=0.005, ref_price=50000)
buy_levels, sell_levels = generate_grid_levels(
    anchor_price=50000, buy_spacing=spacing, sell_spacing=spacing, grid_levels_count=10)
```

```python
# Avellaneda-Stoikov only
from core.inventory import calculate_reservation_price, calculate_optimal_spread, get_skewed_grid_params

r = calculate_reservation_price(mid_price=50000, inventory_q=0.3, gamma=0.5, volatility=0.002)
delta = calculate_optimal_spread(volatility=0.002, gamma=0.5, kappa=1.5)
buy_sp, sell_sp, anchor = get_skewed_grid_params(
    mid_price=50000, inventory_q=0.3, gamma=0.5, volatility=0.002, kappa=1.5,
    time_horizon=96.0)  # T=96 = 1 day; produces ~$2 skew at full inventory
```

```python
# Risk calculations only
from core.risk import (
    get_margin_tier, calculate_maintenance_margin,
    calculate_liquidation_price, calculate_var_95, check_var_constraint
)

mmr, cum = get_margin_tier(100000)  # $100K notional
liq = calculate_liquidation_price(entry_price=50000, position_size=0.1,
                                   wallet_balance=1000, side=1, leverage=5.0)
var = calculate_var_95(portfolio_value=50000, sigma_15m=0.002)
blocked = check_var_constraint(50000, 0.002, 0.15, 10000)
```

---

## Running Tests

### Run All Tests

```bash
python3 -m pytest tests/ -v
```

### Run Specific Test Modules

```bash
# Indicator tests (KAMA, ER, ATR, Z-Score, regime)
python3 -m pytest tests/test_indicators.py -v

# Strategy integration tests (backtest, hedge mode, lookahead)
python3 -m pytest tests/test_strategy.py -v

# Risk management tests (margin, liquidation, VaR)
python3 -m pytest tests/test_risk.py -v

# Pruning tests (all 5 methods)
python3 -m pytest tests/test_pruning.py -v
```

### Run a Single Test

```bash
python3 -m pytest tests/test_strategy.py::TestCircuitBreaker::test_halts_on_crash -v
```

### Run with Output

```bash
# Show print statements
python3 -m pytest tests/ -v -s

# Stop on first failure
python3 -m pytest tests/ -v -x
```

### Alternative: Run Test Files Directly

```bash
python3 tests/test_indicators.py
python3 tests/test_strategy.py
python3 tests/test_risk.py
python3 tests/test_pruning.py
```

---

## Configuration Reference

All parameters live in `config.py` in three dictionaries:

### Advanced Pipeline Features (V4.2)

#### 1. Machine Learning State Classification (GMM)
By default, the strategy discovers market regimes using a Kaufman Adaptive Moving Average (KAMA) FSM. You can switch to **Unsupervised Machine Learning**:
```python
# In config.py -> STRATEGY_PARAMS
'use_ml_regime': True 
```
When enabled, a Gaussian Mixture Model clusters rolling returns and normalized volatility into Bull, Bear, and Noise states dynamically.

#### 2. ADX Veto Filter (new in V4.2)
All trend signals (UPTREND / DOWNTREND) from both KAMA and GMM are vetoed back to NOISE when `ADX < adx_trend_threshold` (default 25). This prevents the strategy from taking one-sided positions during whale fakeouts — a critical crypto-specific failure mode.
```python
'adx_period': 14
'adx_trend_threshold': 25.0  # raise to 30 for more conservative trend detection
```

#### 3. Geometric Grid with Round-Number Avoidance (new in V4.2)
Grids default to percentage-based spacing that widens exponentially outward instead of linear arithmetic spacing. Additionally, levels landing near round-number stop-hunt magnets (e.g., $60,000) are nudged 0.15% away.
```python
'grid_mode': 'geometric'  # default; set to 'arithmetic' for legacy behavior
```

#### 4. Quarter-Kelly Dynamic Sizing (new in V4.2)
Order size is dynamically scaled by a Quarter-Kelly fraction computed from the last 50 regime-filtered trades. This adapts exposure to current strategy edge rather than using a fixed `order_pct`.
```python
'kelly_window': 50    # rolling trade window
'order_pct': 0.05     # base size; Kelly multiplier scales this down
```

#### 5. Stop-Loss Cooldown + Candle-Close Stage 1 (new in V4.2)
- **Candle-Close Stops**: Stage 1 partial stop-loss only fires when the *closing price* crosses the stop level. This prevents flash crash wicks from triggering premature exits that recover within the same bar.
- **Cooldown De-scaling**: After 2 consecutive stops within 48 bars (12h), `order_pct` is halved for the next 48 bars to prevent re-entering at full size during liquidation cascades.
```python
'stop_cooldown_bars': 48
'stop_cooldown_thresh': 2
```

#### 6. VaR Pre-emptive De-Leverage (new in V4.2)
At 75% of the `max_drawdown_pct` limit, the strategy force-closes the single worst fill on each side (tagged `PRUNE_VAR_WARNING`) before the hard cap is triggered. Gradual de-leveraging is less disruptive than a sudden full halt.

#### 7. Weekend / Low-Liquidity De-Scaling (new in V4.2)
When current bar volume is below 50% of the rolling 7-day average, `max_position_pct` is automatically reduced by 30%. No external calendar needed — uses existing OHLCV volume data.
```python
'low_volume_threshold': 0.5  # 50% of 7-day rolling average
```

#### 8. Funding Harvesting Sizing (new in V4.2)
In NOISE regime, when `|funding_rate| > 0.02%`, order size on the receiving side is increased 25% and reduced 25% on the paying side. This harvests the perp funding premium without directional betting.
```python
'funding_harvest_threshold': 0.0002  # 0.02%
```

#### 9. Asymmetric Grids & Real Funding Rates
- **Asymmetric Grids**: Spacing adapts based on Efficiency Ratio (trend strength).
- **Historical Funding**: Real 8h funding rates from Binance are interpolated into 15m Parquet files for precise holding cost simulation.

### STRATEGY_PARAMS

The main strategy configuration. See [Parameter Reference in README.md](README.md#parameter-reference) for the full table.

### BACKTEST_CONFIG

Default backtest settings used when arguments are not provided:

```python
BACKTEST_CONFIG = {
    'coins': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    'start_date': '2024-01-01',
    'end_date': None,        # None = current date
    'timeframe': '15m',
}
```

### OPTIMIZER_SPACE

Defines the parameter search space for Optuna. Parameters are grouped by category for clarity. Commented-out entries are fixed at `STRATEGY_PARAMS` defaults.

To add a new parameter to optimization:

```python
OPTIMIZER_SPACE = {
    # ... existing entries ...
    'my_new_param': {'type': 'float', 'low': 0.1, 'high': 5.0},
}
```

To remove a parameter from optimization, comment out its entry.

---

## Data Management

### Cache

OHLCV data is automatically cached as Parquet files in `data/cache/`:

```
data/cache/BTCUSDT_15m_1704067200000_1739491200000.parquet
```

The filename includes the symbol, timeframe, and timestamp range. If the exact file exists, it's loaded from disk instead of re-fetching from the exchange.

### Clearing Cache

```bash
# Remove all cached data
rm -rf data/cache/

# Remove cache for a specific coin
rm data/cache/BTCUSDT_*.parquet
```

### First-Time Data Fetch

On first run for a coin/date range, the system fetches data from Binance using ccxt. This may take several minutes depending on the date range. Progress is printed to console.

**Note**: Binance API has rate limits. The fetcher has built-in rate limiting (`enableRateLimit: True`) and retry logic on failures.

---

## Output Files

### Trade Logs (CSV)

Location: `data/output/{COIN}_trades.csv`

| Column | Description |
|--------|-------------|
| `bar` | Bar index in the backtest |
| `timestamp` | Unix timestamp (ms) |
| `datetime` | Human-readable datetime |
| `price` | Execution price |
| `qty` | Trade quantity |
| `label` | Trade type label (e.g., `BUY_OPEN_LONG`) |
| `regime` | Active regime at time of trade |
| `pnl` | Realized PnL from this trade |

### Charts (PNG)

Location: `data/charts/`

- `{COIN}_equity.png` — Equity curve (top) + drawdown chart (bottom)
- `{COIN}_trades.png` — Trade map with color-coded markers:
  - Green triangles: Buy open long
  - Blue triangles: Sell close long
  - Red triangles: Sell open short
  - Orange triangles: Buy close short
  - Dark red: Stops
  - Purple: Circuit breaker
  - Black: Liquidation

### Optimizer Output

| File | Contents |
|------|----------|
| `data/optimizer_results_robust.json` | **Primary output** — robust params, param stability (CV), robustness score, OOS/MC results |
| `data/best_params_v4.json` | Single-best trial params (reference only, may overfit) |
| `grid_v4_wfo.db` | Optuna study database (SQLite, all trials stored) |

---

## Common Workflows

### Workflow 1: First-Time Backtest

```bash
pip install -r requirements.txt
python3 main.py --coins BTC --start 2024-01-01
# Review console output, charts in data/charts/, trades in data/output/
```

### Workflow 2: Optimize and Apply (Robust)

```bash
# Step 1: Run optimizer
python3 optimizer.py --coins BTC ETH SOL --trials 200

# Step 2: Preview robust parameter changes
python3 apply_optimized_params.py \
    --params data/optimizer_results_robust.json \
    --key robust_params --preview

# Step 3: Apply if satisfied
python3 apply_optimized_params.py \
    --params data/optimizer_results_robust.json \
    --key robust_params

# Step 4: Re-run backtest with optimized params
python3 main.py --coins BTC ETH SOL --start 2024-01-01
```

### Workflow 3: Compare Long-Only vs Hedge Mode

```bash
# Hedge mode (default)
python3 main.py --coins BTC --start 2024-01-01

# Long-only mode
python3 main.py --coins BTC --start 2024-01-01 --no-short
```

### Workflow 4: Sensitivity Analysis (Script)

```python
# sensitivity_test.py
from engine.strategy import GridStrategyV4
from backtest.data_fetcher import fetch_data
from config import STRATEGY_PARAMS

df = fetch_data('BTC/USDT', '15m', '2024-01-01')

print(f"{'Gamma':<10} {'Return':>10} {'MaxDD':>10} {'Sharpe':>10} {'Calmar':>10}")
print("-" * 50)

for gamma in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    config = STRATEGY_PARAMS.copy()
    config['gamma'] = gamma
    strat = GridStrategyV4(config)
    r = strat.run(df)
    m = r['metrics']
    print(f"{gamma:<10.1f} {m['total_return_pct']:>+9.2f}% "
          f"{m['max_drawdown_pct']:>9.2f}% "
          f"{m['sharpe_ratio']:>10.3f} "
          f"{m['calmar_ratio']:>10.3f}")
```

### Workflow 5: Verify No Bugs After Changes

```bash
python3 -m pytest tests/ -v
# All 57 tests should pass
```

---

## Troubleshooting

### "No module named 'ccxt'"
```bash
pip install -r requirements.txt
```

### "No module named 'pytest'"
```bash
pip install pytest
```

### "Insufficient data for {coin}"
The strategy requires at least 100 candles. Check:
- Is the date range too short?
- Is the symbol correct? Use `BTC` (auto-converted to `BTC/USDT`) or `BTC/USDT`
- Is Binance accessible from your network?

### Data fetch takes too long
First-time fetches for long date ranges can take minutes. After first fetch, data is cached as Parquet and subsequent runs load instantly.

### "KeyError: 'timestamp'" or "KeyError: 'close'"
The DataFrame must have columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`. If using custom data, ensure these column names match exactly.

### Circuit Breaker triggers too often
Lower the absolute value of `halt_z_threshold` in `config.py`:
```python
'halt_z_threshold': -4.0,   # Was -3.0, now requires stronger crash
```

### No trades happening
Check:
1. Is `grid_levels` > 0?
2. Is `order_pct` > 0?
3. Is `grid_spacing_k` reasonable? (Too large = grid too wide to fill)
4. Is the data series long enough for warmup? (Need ~50 bars for indicators)

### Optimizer produces poor results
- Increase `--trials` (minimum 100 recommended)
- Ensure sufficient data (at least 6 months per coin)
- Try different `--windows` values (2-5)
- Delete `grid_v4_wfo.db` and start fresh if previous study was with different data
- Check the **Robustness Score** printed at the end — score > 0.50 means the parameter space is too wide

### Too many PRUNE events in backtest output
- `PRUNE_DEVIANCE` often: `deviance_sigma` is too tight — try increasing to 3.0+
- `PRUNE_OLDEST` often: positions not resolving in time — try increasing `max_position_age_hours`
- `PRUNE_GAP` often: `gap_prune_mult` is too small — try increasing to 3.0+
- `PRUNE_FUNDING` often: `funding_cost_ratio` is too low — try increasing to 0.7+

### Charts not generating
- Check `matplotlib` is installed: `pip install matplotlib`
- If running headless (no display), `matplotlib.use('Agg')` is already set in `main.py`
- Charts are saved to `data/charts/`, not displayed interactively

---

**Version**: V4.1 | **Date**: 2026-02-19
