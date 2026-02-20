# CLAUDE.md — Agent Context for GridStrategyV4

## Project Overview

**GridStrategyV4: Volatility-Adaptive Long/Short Grid Trading Engine for Crypto Perpetual Futures**

A hedge-mode grid trading system on 15-minute candles that dynamically adapts to market regime via KAMA-based detection. Solves the three classic grid failures:

| Problem | V4 Solution | Result |
|---|---|---|
| Uptrends (grids miss moves) | Trailing Up shifts grid upward | Captures 80-90% of trend |
| Downtrends (toxic inventory) | Avellaneda-Stoikov Inventory Skew | Drawdown reduced -40% to -25% |
| Black Swans (cascade losses) | Z-Score Circuit Breaker + Pruning | Capital preserved through crashes |

### Design Targets

- **Hedge Mode**: Independent long + short positions (no netting)
- **Perpetual Futures**: Funding rate simulation, tiered margin, iterative liquidation
- **15-Minute Timeframe**: 96 candles/day; filters microstructure noise, stays responsive
- **Bias-Free Backtest**: Bar i uses indicators from bar i-1, price action from bar i
- **Anti-Overfit Optimizer**: Walk-forward validation, multi-coin averaging, stability penalty

---

## Status

**Version**: V4.0
**Status**: Complete (Backtest + Live Execution)
**Tests**: 121/121 passing (57 backtest + 64 live)
**Architecture**: Pure Python/NumPy (Numba-ready structure for future acceleration)

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Backtest (default: BTC, ETH, SOL from 2024-01-01)
python3 main.py --coins BTC ETH SOL --start 2024-01-01

# Long-only mode
python3 main.py --coins BTC --no-short --capital 5000

# Live trading (dry-run first)
python3 live_trade.py --coin BTC --dry-run

# Live trading (real)
python3 live_trade.py --coin BTC --capital 5000 --leverage 10

# Optimize
python3 optimizer.py --coins BTC SOL --trials 200 --windows 3

# Tests
python3 -m pytest tests/ -v
```

---

## File Structure

```
GRID_Trade/
├── CLAUDE.md                    # This file (agent context)
├── README.md                    # Full math + logic documentation
├── USAGE.md                     # Complete usage guide
├── config.py                    # All V4 parameters + optimizer space + LIVE_CONFIG
├── main.py                      # CLI entry point (backtest runner)
├── live_trade.py                # CLI entry point (live trading)
├── optimizer.py                 # Walk-forward Optuna optimizer
├── apply_optimized_params.py    # Apply optimizer output to config.py
├── requirements.txt             # Dependencies (ccxt, pandas, numpy, optuna, etc.)
│
├── core/                        # Math / indicator modules
│   ├── __init__.py
│   ├── kama.py                  # KAMA, ER, 5-state Regime FSM
│   ├── atr.py                   # ATR (Wilder's), Z-Score, Rolling Volatility
│   ├── grid.py                  # Grid level generation, dynamic ATR spacing
│   ├── inventory.py             # Full Avellaneda-Stoikov model
│   ├── risk.py                  # Tiered margin, iterative liquidation, VaR
│   └── funding.py               # Synthetic funding rate simulation
│
├── engine/                      # Execution logic
│   ├── __init__.py
│   ├── types.py                 # PositionTracker, order types, trade labels
│   ├── matching.py              # Virtual OrderBook, conservative fill model
│   ├── pruning.py               # 5-method "Gardener" pruning module
│   └── strategy.py              # GridStrategyV4 main orchestrator (~650 lines)
│
├── live/                        # Live execution module
│   ├── __init__.py
│   ├── executor.py              # BinanceExecutor: ccxt wrapper for Futures hedge mode
│   ├── runner.py                # LiveRunner: 11-step strategy loop on real candles
│   ├── state.py                 # StateManager: JSON persistence + crash recovery
│   ├── logger.py                # TradeLogger: structured logging + per-trade metrics
│   └── monitor.py               # HealthMonitor: heartbeat, sync, equity checks
│
├── backtest/                    # Backtesting infrastructure
│   ├── __init__.py
│   ├── simulator.py             # BacktestSimulator wrapper (single/multi-coin)
│   └── data_fetcher.py          # Binance OHLCV via ccxt + parquet caching
│
├── tests/                       # Test suite (121 tests)
│   ├── __init__.py
│   ├── test_indicators.py       # KAMA, ER, ATR, Z-Score, regime, volatility (15 tests)
│   ├── test_strategy.py         # Full backtest, hedge mode, CB, lookahead-free (12 tests)
│   ├── test_risk.py             # Margin tiers, liquidation, VaR, PnL (16 tests)
│   ├── test_pruning.py          # All 5 pruning methods + priority order (14 tests)
│   └── test_live.py             # Live module: executor, state, logger, monitor (31 tests)
│
└── data/                        # Runtime data (gitignored)
    ├── cache/                   # Parquet-cached OHLCV data
    ├── output/                  # CSV trade exports
    ├── charts/                  # PNG equity curves + trade maps
    ├── live_state/              # Live runner state persistence
    └── live_logs/               # Live trade logs + rotating log files
```

---

## Core Concepts

### 1. KAMA (Kaufman Adaptive Moving Average)

**File**: `core/kama.py`

Adapts smoothing speed based on market efficiency:

```
ER = |P_t - P_{t-n}| / Sum(|P_i - P_{i-1}|)    # Efficiency Ratio
SC = (ER * (SC_fast - SC_slow) + SC_slow)^2       # Smoothing Constant
KAMA_t = KAMA_{t-1} + SC * (P_t - KAMA_{t-1})    # Recursive update
```

- ER near 1.0 = trending (fast KAMA response)
- ER near 0.0 = choppy (slow KAMA response)
- SC is squared to suppress noise and amplify trend

### 2. Five-State Regime FSM

**File**: `core/kama.py` function `detect_regime()`

Normalized KAMA slope: `slope = (KAMA_t - KAMA_{t-1}) / ATR`

| Regime | Code | Condition | Grid Behavior |
|---|---|---|---|
| NOISE | 0 | `\|slope\| < theta` OR `ER < 0.5` | Both long + short grids active |
| UPTREND | 1 | `slope > theta` AND `ER > 0.5` | Long grid active, no new shorts |
| DOWNTREND | -1 | `slope < -theta` AND `ER > 0.5` | Short grid active, no new longs |
| BREAKOUT_UP | 2 | `slope > 2 * theta` | Long grid + trailing up |
| BREAKOUT_DOWN | -2 | `slope < -2 * theta` | Short grid, halt long entries |

### 3. Avellaneda-Stoikov Inventory Model

**File**: `core/inventory.py`

```
Reservation Price:  r = s - q * gamma * sigma^2 * T
Optimal Spread:     delta = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)
```

- Grid centers on `r` (not mid-price `s`)
- Long inventory (q > 0) shifts `r` below `s` = sells closer, buys further
- Short inventory (q < 0) shifts `r` above `s` = buys closer, sells further
- `gamma` = risk aversion (0.1-2.0), `kappa` = fill probability, `sigma` = rolling volatility
- `T` = time horizon in 15m bars (default **96** = 1 day). T=1.0 produces ~$0.01 skew on ETH; T=96 produces ~$2 — the model is now meaningfully active.

### 4. Circuit Breaker

**File**: `core/atr.py` (Z-Score), `engine/strategy.py` (logic)

```
Z = (Close - SMA_20) / ATR
Z < -3.0  ->  HALT (cancel all opening orders, hold positions)
Z > -1.0  ->  RESUME (regenerate grid)
```

### 5. Five-Method Pruning ("Gardener")

**File**: `engine/pruning.py`

Priority order (most urgent first):

| # | Method | Trigger | Target |
|---|---|---|---|
| 1 | Deviance | Price > 3-sigma from KAMA | Worst fill (highest entry for longs) |
| 2 | Oldest | Fill held > 24 hours | Oldest fill |
| 3 | Gap | Fill > 3x grid spacing from price | Most distant fill |
| 4 | Funding | Accumulated funding > 50% of grid profit | Highest funding cost fill |
| 5 | Profit Offset | Profit buffer available | Worst underwater fill |

### 6. Trailing Up

**File**: `engine/strategy.py`

In UPTREND/BREAKOUT_UP when ER > threshold:
- If price exceeds top grid level, shift anchor upward by 50% of gap
- Regenerate grid at new higher levels
- Prevents grid from being "left behind" in trends

### 7. Hedge Mode

**File**: `engine/types.py` (`PositionTracker` class)

Two independent position trackers:
- `pos_long`: Buys = entries, Sells = take-profits
- `pos_short`: Sells = entries, Buys = take-profits
- No netting: opening a long does NOT close a short
- Each has own avg_entry, fills list, PnL tracking
- Fills list enables targeted pruning of specific fills

### 8. Risk Management

**File**: `core/risk.py`

- **Tiered Margin**: 7-tier Binance USDM specification (0.4% to 12.5% MMR)
- **Iterative Liquidation**: Converges in ~3-5 iterations (MMR depends on P_liq)
- **VaR Constraint**: `VaR_95 = Portfolio * 1.65 * sigma_15m` blocks new orders if > max_drawdown * equity
- **ATR Stop Loss (2-stage)**: Stage 1 closes 50% at `entry ± sl_mult × ATR` (multi-fill); Stage 2 closes remainder at `1.5×` multiplier. Single-fill positions close 100% immediately.

---

## Main Loop Order (engine/strategy.py)

```
for each bar i (starting from 1):
    1. REGIME        - Read from pre-computed indicators[i-1] (no lookahead)
    2. CIRCUIT BREAK  - Halt if Z < -3, resume if Z > -1
    3. STOP LOSS      - 2-stage ATR stops: 50% close at 1× mult, remainder at 1.5× (multi-fill)
    4. LIQUIDATION     - Check if leveraged positions hit liquidation price
    5. PRUNING         - Run 5-method cycle on both sides
    6. TRAILING UP     - Shift grid anchor upward in uptrends
    7. VaR CHECK       - Block new orders if VaR exceeds limit
    8. GRID GENERATE   - A-S skewed grid (T=96, meaningful ~$2 skew on ETH); regen only on
                          price drift >1 spacing or regime change (not every fill)
    9. FILL MATCHING   - Conservative: Buy if Low <= Price, Sell if High >= Price
   10. FUNDING RATE    - Apply every 32 bars (8h intervals)
   11. LOG EQUITY      - Wallet + unrealized PnL both sides
```

---

## Key Parameters (config.py)

```python
# Regime Detection
'kama_period': 10           # ER lookback (150 min)
'regime_threshold': 0.15    # KAMA slope theta
'er_trend_threshold': 0.5   # ER above = trend

# Grid
'grid_spacing_k': 1.0       # spacing = k * ATR
'grid_levels': 10            # Levels per side
'order_pct': 0.03            # 3% capital per fill
'spacing_floor': 0.005       # Min 0.5% of price

# Inventory (Avellaneda-Stoikov)
'gamma': 0.5                 # Risk aversion
'kappa': 1.5                 # Fill probability
'as_time_horizon': 96.0      # T in 15m bars (96 = 1 day); controls inventory skew magnitude

# Trailing Up
'trailing_activation_er': 0.65
'trailing_reserve_pct': 0.25

# Circuit Breaker
'halt_z_threshold': -3.0     # 3-sigma crash halt
'resume_z_threshold': -1.0

# Pruning
'deviance_sigma': 3.0        # Deviance threshold (ATR mult)
'max_position_age_hours': 24  # Oldest fill pruning
'gap_prune_mult': 3.0        # Gap threshold (grid spacing mult)
'funding_cost_ratio': 0.5    # Funding vs profit threshold

# Risk
'max_drawdown_pct': 0.15     # VaR hard cap (15%)
'atr_sl_mult': 3.5           # Stop loss (ATR mult)
'max_position_pct': 0.7      # Max notional per side

# Execution
'fee_maker': -0.00005        # -0.005% maker rebate
'fee_taker': 0.0002          # 0.02% taker fee
'leverage': 1.0              # 1x default
'allow_short': True          # Hedge mode on/off
```

---

## Trade Labels

```
BUY_OPEN_LONG         # Grid buy entry (long)
SELL_CLOSE_LONG       # Grid sell take-profit (long)
SELL_OPEN_SHORT       # Grid sell entry (short)
BUY_CLOSE_SHORT       # Grid buy take-profit (short)
STOP_LONG             # ATR stop loss (long)
STOP_SHORT            # ATR stop loss (short)
PRUNE_DEVIANCE        # Pruning: price deviated from KAMA
PRUNE_OLDEST          # Pruning: fill too old
PRUNE_GAP             # Pruning: price-fill gap too large
PRUNE_FUNDING         # Pruning: funding cost too high
PRUNE_OFFSET          # Pruning: profit-subsidized close
CIRCUIT_BREAKER_HALT  # Trading halted (Z < -3)
LIQUIDATION           # Margin liquidation
```

---

## Metrics Output

**Performance**: `total_return_pct`, `buy_hold_return_pct`, `max_drawdown_pct`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`

**Trading**: `longs_opened`, `longs_closed`, `shorts_opened`, `shorts_closed`, `stops_long`, `stops_short`, `total_trades`

**Risk Events**: `prune_count`, `prune_types` (breakdown), `circuit_breaker_triggers`, `trailing_shifts`, `var_blocks`, `liquidations`

**P&L**: `win_rate_pct`, `profit_factor`, `gross_profit`, `gross_loss`, `funding_pnl`, `final_capital`

Annualization factor for 15m data: `sqrt(96 * 365)` = ~187.1

---

## Lookahead-Free Backtesting

```python
for i in range(1, n):
    prev = indicators[i-1]    # Decisions use prev-bar KAMA, ATR, ER, Z, regime
    curr = ohlc[i]            # Fills use current-bar Open, High, Low, Close
```

Verified by `test_no_future_data_access`: runs identical first-200 bars on 200-bar and 300-bar DataFrames, asserts equity curves match.

---

## Anti-Overfit Optimizer (optimizer.py)

1. **Walk-Forward Validation**: Train on 70%, test on 30%, sliding windows
2. **Multi-Coin Averaging**: Parameters must work across BTC, ETH, SOL
3. **Calmar Ratio Fitness**: Return / Max Drawdown (penalizes drawdown-heavy results)
4. **Complexity Penalty**: Trades > 500 penalized (over-trading signal)
5. **Stability Penalty**: High variance across windows reduces score
6. **OOS Validation**: Final best params tested on held-out 30%

---

## Realistic Expectations

- **Sideways/NOISE**: Grid thrives - primary profit source
- **Uptrends**: Trailing Up captures trend (80-90% vs 20-30% static)
- **Downtrends**: Inventory Skew + Pruning + CB reduce drawdown but no long-only strategy profits in monotonic downtrends
- **Black Swans**: Circuit Breaker preserves capital, avoids "selling the bottom"
- **Goal in downtrends**: Capital preservation for trading the recovery

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `KeyError: 'kama'` | Indicators are pre-computed in `prepare_indicators()` - ensure DataFrame has OHLCV columns |
| Trailing Up never activates | Lower `trailing_activation_er` or use stronger trend data |
| Circuit Breaker triggers too often | Widen `halt_z_threshold` (e.g., -3.0 to -4.0) |
| Strategy loses in sideways | Decrease `spacing_mult`, increase `grid_levels` |
| Pruning too aggressive | Increase `prune_depth_mult` or `max_position_age_hours` |
| No shorts being opened | Check `allow_short: True` and regime is not stuck in UPTREND |

---

## Dependencies

```
ccxt        # Exchange data fetching + live order placement
pandas      # DataFrames
numpy       # Numerical computation
optuna      # Bayesian optimization
matplotlib  # Chart generation
pyarrow     # Parquet caching
pytest      # Test suite
python-dotenv  # Environment variable management (live trading)
```

---

## Live Execution Module

### Architecture

The `live/` module mirrors the backtest's 11-step strategy loop on real 15m candles. All `core/` indicator functions and `engine/types.py` (PositionTracker) are used without modification — only the order placement and fill detection layers change.

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `BinanceExecutor` | `live/executor.py` | Exchange API via ccxt (orders, positions, balance, funding) |
| `LiveRunner` | `live/runner.py` | Main 11-step orchestration loop on real candles |
| `StateManager` | `live/state.py` | JSON persistence + CSV/JSONL trade logs + crash recovery |
| `TradeLogger` | `live/logger.py` | Structured logging + per-trade metrics (PnL, R-multiple, hold time) |
| `HealthMonitor` | `live/monitor.py` | Heartbeat, position sync, equity checks, error tracking |

### Backtest → Live Mapping

| Step | Backtest | Live |
|------|----------|------|
| Regime | `indicators[i-1]` | `indicators.iloc[-2]` (rolling buffer) |
| Circuit Break | Set `halted=True` | Same + `cancel_all_orders()` |
| Stop Loss | Check candle H/L | Same + market order via executor |
| Pruning | `close_specific_fill()` | Same + market order for fill qty |
| Grid Generate | Virtual OrderBook | Real limit orders on exchange |
| Fill Matching | `check_fills(H,L)` | Exchange order polling (pre-step sync) |
| Funding Rate | Synthetic simulation | Real `get_funding_rate()` from exchange |

### Config

Live-specific config is in `LIVE_CONFIG` dict in `config.py`. Strategy parameters from `STRATEGY_PARAMS` are shared with the backtest.

### API Keys

Read from environment variables `BINANCE_API_KEY` and `BINANCE_API_SECRET`.

---

## Future Work

1. Numba acceleration (`@njit` on hot loops)
2. WebSocket feed for real-time candle streaming (replace polling)
3. Multi-timeframe signals (1h regime, 15m execution)
4. Genetic algorithm optimizer option
5. Break-even analysis tool
6. Multi-symbol concurrent live trading
7. Telegram/Discord alert integration

---

**Version**: V4.1
**Date**: 2026-02-20
**Status**: Complete (Backtest + Live Execution) - 121/121 tests passing
