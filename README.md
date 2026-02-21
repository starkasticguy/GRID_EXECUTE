# GridStrategyV4 — Volatility-Adaptive Grid Trading Engine

A hedge-mode grid trading system for cryptocurrency perpetual futures on 15-minute candles. V4.2 adds 9 crypto-native hardening improvements on top of V4's KAMA-based regime detection and Avellaneda-Stoikov inventory management: ADX veto filter, stop-loss cooldown, candle-close stops, geometric grid with round-number avoidance, Quarter-Kelly sizing, VaR pre-emptive de-leverage, weekend de-scaling, funding harvesting sizing, and portfolio correlation VaR.

## Why This Exists

Traditional grid strategies fail in three predictable ways:

1. **In uptrends**, they sell too early and miss 70-80% of the move
2. **In downtrends**, they accumulate toxic inventory that bleeds the account
3. **In crashes**, they provide liquidity into the abyss (buying the falling knife)

V4 addresses each failure with a dedicated mechanism.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Signal Layer: How the Strategy Reads the Market](#signal-layer-how-the-strategy-reads-the-market)
  - [Efficiency Ratio (ER)](#efficiency-ratio-er)
  - [KAMA (Kaufman Adaptive Moving Average)](#kama-kaufman-adaptive-moving-average)
  - [ATR (Average True Range)](#atr-average-true-range)
  - [ADX (Average Directional Index)](#adx-average-directional-index)
  - [Regime Detection (Finite State Machine)](#regime-detection-finite-state-machine)
  - [Z-Score (Circuit Breaker Signal)](#z-score-circuit-breaker-signal)
  - [Rolling Volatility](#rolling-volatility)
- [Grid Layer: How Orders Are Placed](#grid-layer-how-orders-are-placed)
  - [Dynamic Spacing](#dynamic-spacing)
  - [Grid Level Generation](#grid-level-generation)
  - [Order Sizing (Quarter-Kelly)](#order-sizing-quarter-kelly)
  - [Avellaneda-Stoikov Inventory Skew](#avellaneda-stoikov-inventory-skew)
  - [Funding Rate Bias and Harvesting](#funding-rate-bias-and-harvesting)
- [Execution Layer: How Trades Happen](#execution-layer-how-trades-happen)
  - [Hedge Mode (Dual Positions)](#hedge-mode-dual-positions)
  - [Virtual Order Book](#virtual-order-book)
  - [Fill Model](#fill-model)
  - [Trade Labels](#trade-labels)
- [Protection Layer: How the Strategy Defends Capital](#protection-layer-how-the-strategy-defends-capital)
  - [Circuit Breaker](#circuit-breaker)
  - [ATR Stop Loss (with Candle-Close Anti-Wick + Cooldown)](#atr-stop-loss)
  - [Tiered Margin and Liquidation](#tiered-margin-and-liquidation)
  - [VaR Constraint + Pre-emptive De-Leverage](#var-constraint)
  - [Weekend / Low-Liquidity De-Scaling](#weekend--low-liquidity-de-scaling)
  - [Trailing Up](#trailing-up)
  - [Five-Method Pruning](#five-method-pruning)
- [Main Loop: Step-by-Step Execution](#main-loop-step-by-step-execution)
- [Backtesting: How We Avoid Lies](#backtesting-how-we-avoid-lies)
- [Optimizer: How Parameters Are Chosen](#optimizer-how-parameters-are-chosen)
- [Metrics: What We Measure](#metrics-what-we-measure)
- [Parameter Reference](#parameter-reference)
- [Testing](#testing)
- [Realistic Expectations](#realistic-expectations)

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `ccxt`, `pandas`, `numpy`, `optuna`, `matplotlib`, `pyarrow`, `pytest`

Python 3.10+ recommended.

---

## Quick Start

```bash
# Default backtest: BTC, ETH, SOL from 2024-01-01
python3 main.py

# Single coin with custom parameters
python3 main.py --coins BTC --start 2023-06-01 --end 2024-06-01 --capital 5000

# Long-only mode (no short grid)
python3 main.py --coins ETH --no-short

# Skip chart generation
python3 main.py --coins SOL --no-plot

# Run optimizer
python3 optimizer.py --coins BTC SOL --trials 200

# Run tests
python3 -m pytest tests/ -v
```

---

## Architecture Overview

The system has four layers, each in its own directory:

```
Signal Layer (core/)          What regime are we in? How volatile is it?
    |
Grid Layer (core/)            Where should orders be placed? How large?
    |
Execution Layer (engine/)     Match orders, track positions, process fills
    |
Protection Layer (engine/)    Stop losses, circuit breaker, pruning, VaR
```

The orchestrator (`engine/strategy.py`) runs all four layers sequentially on every bar.

---

## Signal Layer: How the Strategy Reads the Market

### Efficiency Ratio (ER)

**File**: `core/kama.py` | **Function**: `calculate_er()`

The Efficiency Ratio measures how "straight" price movement is over a window. It answers: "Of all the price movement in the last N bars, how much was directional vs noise?"

**Formula:**

```
           |Close_t - Close_{t-n}|           Net directional movement
ER_t = --------------------------------- = ---------------------------
        Sum_{i=1..n} |Close_i - Close_{i-1}|    Total path length
```

**Interpretation:**
- **ER = 1.0**: Price moved in a perfectly straight line (pure trend). Every tick was in the same direction.
- **ER = 0.0**: Price went nowhere despite lots of movement (pure noise). The market chopped back and forth.
- **ER = 0.5**: Half the movement was directional, half was noise.

**Why it matters**: ER tells us whether grid trading will work. Grids profit from mean reversion (low ER). When ER is high, price is trending and grids accumulate losing inventory.

**Default**: `kama_period = 10` (150 minutes on 15m candles)

**Warmup**: First `period` values are set to 0 (insufficient data).

---

### KAMA (Kaufman Adaptive Moving Average)

**File**: `core/kama.py` | **Function**: `calculate_kama()`

KAMA is a moving average that adapts its speed based on market conditions. In trends, it tracks price closely. In chop, it barely moves.

**Formula (3 steps):**

**Step 1** — Smoothing Constant (SC):
```
SC_fast = 2 / (fast_period + 1)     # Fast EMA constant (default: 2/(2+1) = 0.667)
SC_slow = 2 / (slow_period + 1)     # Slow EMA constant (default: 2/(30+1) = 0.0645)

SC_t = (ER_t * (SC_fast - SC_slow) + SC_slow)^2
```

The squaring is critical: it makes the transition between fast and slow non-linear. When ER is low, SC becomes very small (KAMA barely moves). When ER is high, SC approaches SC_fast^2 (KAMA tracks quickly).

**Step 2** — Recursive update:
```
KAMA_t = KAMA_{t-1} + SC_t * (Close_t - KAMA_{t-1})
```

This is an exponential moving average where the smoothing factor changes every bar.

**Step 3** — KAMA slope for regime detection:
```
slope_t = (KAMA_t - KAMA_{t-1}) / ATR_t
```

Normalizing by ATR makes the slope comparable across assets and time periods.

**Why it matters**: KAMA provides both direction (slope sign) and conviction (slope magnitude) in one indicator. Traditional approaches need separate trend strength (ADX) and direction indicators.

---

### ATR (Average True Range)

**File**: `core/atr.py` | **Function**: `calculate_atr()`

ATR measures the average bar-to-bar price range, accounting for gaps.

**True Range:**
```
TR_t = max(High_t - Low_t,  |High_t - Close_{t-1}|,  |Low_t - Close_{t-1}|)
```

The three components handle:
1. Normal intrabar range (High - Low)
2. Gap up then sell-off (High far above previous close)
3. Gap down then rally (Low far below previous close)

**Smoothing (Wilder's EMA):**
```
alpha = 2 / (period + 1)
ATR_t = alpha * TR_t + (1 - alpha) * ATR_{t-1}
```

**Default**: `atr_period = 14` (3.5 hours on 15m candles)

**Used for**: Grid spacing, stop losses, regime slope normalization, Z-Score denominator.

---

### ADX (Average Directional Index)

**File**: `core/adx.py` | **Function**: `calculate_adx()`

ADX measures trend *strength* on a scale of 0-100, but unlike KAMA it has no directional bias. It answers: "Is the market actually trending, or is a directional move just noise?"

**Calculation (Wilder's smoothing):**

```
+DM = max(High_t - High_{t-1}, 0) if > abs(Low_t - Low_{t-1}), else 0
-DM = max(Low_{t-1} - Low_t, 0) if > abs(High_t - High_{t-1}), else 0
+DI = 100 * RMA(+DM, period) / ATR
-DI = 100 * RMA(-DM, period) / ATR
DX  = 100 * |+DI - -DI| / (+DI + -DI)
ADX = RMA(DX, period)         # Wilder's smoothed DX
```

**Interpretation:**
- `ADX < 20`: No real trend (sideways chop or just starting)
- `ADX 20–25`: Borderline — possible trend forming
- `ADX > 25`: Confirmed trend direction
- `ADX > 40`: Strong trend

**V4.2 use — ADX Veto Filter**: If GMM or KAMA classifies the market as UPTREND or DOWNTREND but `ADX < adx_trend_threshold` (default 25), the regime is **forced to NOISE**. This prevents the strategy from taking one-sided trend positions during high-volatility crypto chop, which looks like a trend on KAMA slope alone but has no genuine directional momentum.

**Default**: `adx_period = 14`, `adx_trend_threshold = 25.0`

---

### Regime Detection (Finite State Machine)

**File**: `core/kama.py` | **Function**: `detect_regime()`

The regime detector is a finite state machine (FSM) that classifies every bar into one of five states:

```
        BREAKOUT_DOWN (-2)
              |
        DOWNTREND (-1)
              |
NOISE (0) ----+---- UPTREND (1)
                      |
                BREAKOUT_UP (2)
```

**Classification logic:**

```python
slope = (KAMA_t - KAMA_{t-1}) / ATR_t

if |slope| > 2 * theta:          # Breakout (strongest signal)
    BREAKOUT_UP if slope > 0
    BREAKOUT_DOWN if slope < 0
elif ER > er_trend_threshold:     # Trend
    if slope > theta:  UPTREND
    elif slope < -theta:  DOWNTREND
    else:  NOISE
else:                              # Low efficiency
    NOISE
```

**How each regime affects the grid:**

| Regime | Long Grid | Short Grid |
|--------|-----------|------------|
| NOISE | Open entries + TP sells | Open entries + TP buys |
| UPTREND | Open entries + TP sells | TP buys only (no new shorts) |
| DOWNTREND | TP sells only (no new longs) | Open entries + TP buys |
| BREAKOUT_UP | Open entries + trailing up | TP buys only (no new shorts) |
| BREAKOUT_DOWN | TP sells only (no new longs) | Open entries + TP buys |

**Default thresholds**: `regime_threshold = 0.15`, `er_trend_threshold = 0.5`

---

### Z-Score (Circuit Breaker Signal)

**File**: `core/atr.py` | **Function**: `calculate_z_score()`

The Z-Score measures how far price has deviated from its recent average, normalized by volatility.

```
Z_t = (Close_t - SMA_20_t) / ATR_t
```

**Interpretation:**
- Z = 0: Price is at the 20-bar average (normal)
- Z = -3: Price has crashed 3 ATRs below average (3-sigma event)
- Z = +3: Price has pumped 3 ATRs above average

**Used for**: Circuit Breaker trigger (halt at Z < -3, resume at Z > -1).

---

### Rolling Volatility

**File**: `core/atr.py` | **Function**: `calculate_rolling_volatility()`

Provides the sigma (standard deviation) parameter for the Avellaneda-Stoikov model.

```
log_returns[j] = ln(Close_j / Close_{j-1})     for j in window
sigma_t = std(log_returns)                       # Standard deviation of log returns
```

This is the raw 15-minute sigma (not annualized). The A-S model uses a rolling time horizon T (default 96 = 1 day of 15m bars) that scales inventory risk to a human-interpretable horizon.

---

## Grid Layer: How Orders Are Placed

### Dynamic Spacing

**File**: `core/grid.py` | **Function**: `calculate_dynamic_spacing()`

Grid spacing adapts to volatility:

```
spacing = max(k * ATR, floor * price)
```

- `k` is the spacing coefficient (default 1.0). Higher = wider grid, fewer fills but safer.
- `floor` is the minimum spacing as a fraction of price (default 0.5%). Prevents the grid from becoming too tight in calm markets.

**Behavior:**
- Volatility expands (ATR rises) -> grid widens -> covers more price range
- Volatility contracts (ATR falls) -> grid tightens -> captures smaller moves

---

### Grid Level Generation

**File**: `core/grid.py` | **Functions**: `generate_geometric_grid_levels()`, `generate_grid_levels()`

V4.2 defaults to **geometric** (percentage-based) spacing (`grid_mode = 'geometric'`):

```
Buy levels:  anchor / (1 + pct%)^i   for i = 1..N
Sell levels: anchor * (1 + pct%)^i   for i = 1..N
```

Geometric spacing naturally widens outer levels — appropriate for crypto where distant reversions are less certain than nearby ones.

**Round-Number Avoidance**: After computing each level, a helper nudges any price landing within 0.1% of a round number (e.g., $60,000, $65,000) outward by 0.15%. These round numbers are known stop-hunt magnets and placing grid orders there leads to premature fills on stop-hunt wicks.

The arithmetic grid (`generate_grid_levels()`) is still available via `grid_mode = 'arithmetic'`.

---

### Order Sizing (Quarter-Kelly)

**File**: `core/kelly.py` + `core/grid.py` | **Function**: `compute_kelly_fraction()`

Order size adapts dynamically to recent strategy edge rather than using a fixed `order_pct`:

```
Kelly = (W% / AvgLoss) - ((1 - W%) / AvgWin)
Quarter-Kelly = Kelly * 0.25           # Crypto safety factor
effective_order_pct = order_pct * max(Quarter-Kelly, 0.25)
```

- Computed from the last `kelly_window` (default 50) regime-filtered trades
- Quarter-Kelly (25%) accounts for crypto's auto-correlated return distribution
- Falls back to `order_pct * 0.5` if trade history is insufficient
- Combined multiplicatively with stop-loss cooldown de-scaling when both are active

Fixed sizing (`calculate_order_qty()`) is unchanged for take-profit quantity calculation.

**Default base**: `order_pct = 0.05` (5%), `kelly_window = 50`

---

### Avellaneda-Stoikov Inventory Skew

**File**: `core/inventory.py`

This is the core mathematical model that makes V4's grid asymmetric. Based on the 2008 paper by Avellaneda and Stoikov on optimal market making.

**The Problem**: A symmetric grid accumulates inventory. If price trends down, a long grid keeps buying and the position grows. The more inventory, the more risk.

**The Solution**: Shift the grid center based on inventory, so that when we hold too much, we become more eager to sell and less eager to buy.

#### Reservation Price

```
r = s - q * gamma * sigma^2 * T
```

Where:
- `s` = current mid-price (or KAMA reference)
- `q` = normalized inventory, ranging from -1 (max short) to +1 (max long)
- `gamma` = risk aversion parameter (0.1 to 2.0)
- `sigma` = rolling standard deviation of 15m log returns
- `T` = time horizon in 15m bars (default 96 = 1 day). Controls how strongly inventory risk is penalized — larger T means bigger skew per unit of inventory.

**How it works**: When `q > 0` (we hold long inventory), `r < s`. The grid anchor drops below market price. This means:
- Buy levels move further below price (we're less eager to add longs)
- Sell levels move closer to price (we're more eager to take profit)

When `q < 0` (we hold short inventory), the opposite happens.

#### Optimal Spread

```
delta = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)
```

Where:
- `kappa` = fill probability parameter (higher = expect more fills = tighter spread)

This gives the minimum rational spread. The grid uses `max(delta, k * ATR, floor * price)` to ensure the spread is never too tight.

#### Inventory Normalization

```
q = clamp(position_size / max_inventory, -1, 1)
```

Position size is divided by max allowed inventory per side, then clamped to [-1, 1] for the A-S model.

---

### Funding Rate Bias and Harvesting

**File**: `core/funding.py` + `engine/strategy.py`

In perpetual futures, funding rates create a carry cost/income. V4.2 handles this in two ways:

**Grid Bias (NOISE regime — spacing adjustment):**
- Positive funding: Widen long grid by 1.2×, tighten short (favor shorts which receive funding)
- Negative funding: Tighten long grid, widen short (favor longs)

**Funding Harvesting Sizing (new in V4.2, NOISE regime only):**
When `|funding_rate| > funding_harvest_threshold` (default 0.02%), order size on the *receiving* side is increased by 25% and reduced by 25% on the *paying* side. This is applied after Kelly sizing, creating a genuine alpha edge from persistent funding skew without directional betting.

```
If funding_rate > 0 (shorts receive):
    short_order_pct = min(order_pct * 1.25, order_pct * 1.5)
    long_order_pct  = max(order_pct * 0.75, order_pct * 0.5)
```

**Funding application:** Every 32 bars (8 hours on 15m candles), funding is applied to all open positions. The cost is distributed across fills for funding-based pruning.

---

## Execution Layer: How Trades Happen

### Hedge Mode (Dual Positions)

**File**: `engine/types.py` | **Class**: `PositionTracker`

V4 runs in hedge mode: long and short positions are completely independent. Opening a long does NOT close any short, and vice versa.

Each `PositionTracker` maintains:
- `size`: Total absolute position size
- `avg_entry`: Weighted average entry price across all fills
- `fills[]`: List of individual fill records (price, qty, timestamp, funding_cost)
- `realized_pnl`: Cumulative realized profit/loss
- `unrealized_pnl`: Mark-to-market on open position

The fills list is critical for pruning. Each pruning method can target a specific fill (the worst one, the oldest one, etc.) rather than closing the entire position.

**PnL calculation:**
```
Long PnL:  (exit_price - entry_price) * qty
Short PnL: (entry_price - exit_price) * qty
```

---

### Virtual Order Book

**File**: `engine/matching.py` | **Class**: `OrderBook`

An in-memory order book that manages all pending grid orders. Orders are tagged with:
- `side`: Buy (1) or Sell (-1)
- `direction`: Long grid (1) or Short grid (-1)
- `grid_level`: Which grid level this order belongs to
- `reduce_only`: Whether this is a take-profit order (can only reduce position)

The order book is regenerated when:
- **A fill occurs AND price has drifted >1 grid spacing from the anchor** (preserves queue position for unfilled levels)
- Regime changes between bars
- Circuit breaker halts/resumes
- Trailing up shifts the anchor
- Stop loss or pruning closes a position

---

### Fill Model

**File**: `engine/matching.py` | **Method**: `OrderBook.check_fills()`

Conservative fill assumption for backtesting:
- **Buy limit order fills** if the candle's Low reaches or crosses below the order price
- **Sell limit order fills** if the candle's High reaches or crosses above the order price

No partial fills. No slippage in matching (slippage is handled at the fee level). This is conservative because real fills might execute at better prices.

---

### Trade Labels

Every trade is tagged with a descriptive label:

| Label | Meaning |
|-------|---------|
| `BUY_OPEN_LONG` | Grid buy entry for long position |
| `SELL_CLOSE_LONG` | Grid sell take-profit on long position |
| `SELL_OPEN_SHORT` | Grid sell entry for short position |
| `BUY_CLOSE_SHORT` | Grid buy take-profit on short position |
| `STOP_LONG` / `STOP_SHORT` | ATR-based stop loss triggered |
| `PRUNE_DEVIANCE` | Position pruned: price deviated from KAMA |
| `PRUNE_OLDEST` | Position pruned: fill too old (stale capital) |
| `PRUNE_GAP` | Position pruned: price-fill gap too large |
| `PRUNE_FUNDING` | Position pruned: funding cost too high |
| `PRUNE_OFFSET` | Position pruned: profit-subsidized close |
| `PRUNE_VAR_WARNING` | Position pruned: pre-emptive de-leverage at 75% drawdown cap |
| `CIRCUIT_BREAKER_HALT` | Trading halted due to crash |
| `LIQUIDATION` | Margin call (leveraged accounts only) |

---

## Protection Layer: How the Strategy Defends Capital

### Circuit Breaker

**Trigger**: Z-Score drops below -3.0 (3-sigma crash).

**Action**:
1. Cancel all opening orders immediately
2. Hold existing positions (don't sell the bottom)
3. Log a `CIRCUIT_BREAKER_HALT` trade record

**Resume**: When Z-Score recovers above -1.0, regenerate the grid and resume normal trading.

**Rationale**: During a crash, every grid buy is catching a falling knife. It's better to pause and let the market find a bottom. The circuit breaker prevents the strategy from providing liquidity into panic selling.

---

### ATR Stop Loss

**File**: `engine/strategy.py` | **Method**: `_check_stop_loss()`

Each side has an independent ATR-based stop using a **two-stage mechanism**:

**Stage 1** (multi-fill positions — closes 50%):
```
Long stop:  avg_entry - atr_sl_mult * ATR    → close 50% of position
Short stop: avg_entry + atr_sl_mult * ATR    → close 50% of position
```

**Stage 2** (remainder or single-fill positions — closes 100%):
```
Long stop:  avg_entry - 1.5 * atr_sl_mult * ATR    → close remaining position
Short stop: avg_entry + 1.5 * atr_sl_mult * ATR    → close remaining position
```

**V4.2 — Candle-Close Stage 1 Stops (Anti-Wick)**: Stage 1 partial stop only fires when the *candle close price* crosses the stop level. Stage 2 still uses the wick (high/low) as a safety net. This prevents premature Stage 1 exits during flash crash wicks that recover fully within a single 15m bar.

**V4.2 — Stop-Loss Cooldown**: After `stop_cooldown_thresh` (default 2) consecutive stops within `stop_cooldown_bars` (default 48 = 12h), `order_pct` is halved for the next 48 bars. This prevents the strategy from re-entering at full size during a liquidation cascade and compounds the de-scaling with the Quarter-Kelly multiplier.

**Default**: `atr_sl_mult = 3.5` (3.5× ATR for first stage, 5.25× for second stage)

---

### Tiered Margin and Liquidation

**File**: `core/risk.py`

V4 implements the Binance USDM 7-tier margin system:

| Tier | Max Notional | MMR | Cumulative |
|------|-------------|-----|------------|
| 1 | $50K | 0.40% | $0 |
| 2 | $250K | 0.50% | $50 |
| 3 | $1M | 1.00% | $1,300 |
| 4 | $5M | 2.50% | $8,800 |
| 5 | $25M | 5.00% | $46,300 |
| 6 | $100M | 10.00% | $171,300 |
| 7 | Unlimited | 12.50% | $296,300 |

**Maintenance Margin:**
```
MM = Notional * MMR - Cumulative
```

**Liquidation Price (iterative solver):**

For longs:
```
P_liq = (WalletBalance - Qty * Entry + Cumulative) / (Qty * (MMR - 1))
```

For shorts:
```
P_liq = (WalletBalance + Qty * Entry + Cumulative) / (Qty * (MMR + 1))
```

Since MMR depends on notional (= P_liq * Qty), which depends on P_liq, this is solved iteratively until convergence (typically 3-5 iterations).

---

### VaR Constraint + Pre-emptive De-Leverage

**File**: `core/risk.py` | **Function**: `check_var_constraint()`

Value at Risk caps total exposure:

```
VaR_95 = Total_Exposure * 1.65 * sigma_15m
```

**Hard rule**: If `VaR_95 > max_drawdown_pct * equity`, no new grid orders are placed.

**V4.2 — Pre-emptive De-Leverage**: When unrealized drawdown reaches 75% of `max_drawdown_pct` (i.e., 11.25% on a 15% cap), the strategy force-closes the single worst-performing fill on each side. This provides a gradual ramp-down rather than a sudden hard stop when the limit is hit.

```
if (initial_capital - current_equity) / initial_capital > max_drawdown_pct * 0.75:
    find worst_fill (largest unrealized loss per side)
    close it at market → tagged PRUNE_VAR_WARNING
```

**Portfolio VaR** (`core/risk.py::compute_portfolio_var`): For multi-coin live trading, computes cross-asset correlation. When BTC/ETH/SOL correlation exceeds 0.80 (crypto panic correlation), treats the entire book as one concentrated position for VaR purposes.

**Default**: `max_drawdown_pct = 0.15` (15% of equity)

---

### Weekend / Low-Liquidity De-Scaling

**File**: `engine/strategy.py`

Crypto weekends and low-volume periods have reduced liquidity, wider spreads, and higher susceptibility to flash crashes. V4.2 auto-detects low-liquidity periods using rolling volume:

```
vol_sma7 = rolling mean of volume over last 7 days (7 × 96 bars)
if current_volume < low_volume_threshold * vol_sma7:
    max_position_pct *= 0.70   # 30% smaller maximum position
```

No external APIs or calendar lookups are needed — it uses existing OHLCV volume data. The threshold `low_volume_threshold = 0.5` triggers when current volume is below 50% of the 7-day moving average.

---

### Trailing Up

**File**: `engine/strategy.py` | Main loop step 6

**The problem**: In a strong uptrend, the grid's buy levels are far below the current price. The strategy stops filling because price has moved away from the grid. A static grid captures maybe 20-30% of a trend move.

**The solution**: When the strategy detects a strong uptrend (ER > `trailing_activation_er` AND regime is UPTREND or BREAKOUT_UP), it checks if price has exceeded the top grid level. If so:

```
shift = (current_price - grid_anchor_long) * 0.5    # Conservative: 50% of gap
grid_anchor_long += shift
regenerate_grid()
```

The 0.5 multiplier makes the shift conservative: the grid moves halfway toward the current price rather than all the way. This avoids buying the exact top.

**Risk mitigation**: `trailing_cap_price` can set an absolute maximum price for the grid anchor, preventing the strategy from chasing unsustainable highs.

---

### Five-Method Pruning

**File**: `engine/pruning.py`

Pruning actively closes individual fills that have become toxic, stale, or uneconomical. Unlike a stop loss (which closes the entire position), pruning surgically removes specific fills.

**Priority order** (run in sequence, first match wins):

#### Method 1: Deviance Pruning (Toxic Position)
```
Long:  prune if price < KAMA - sigma_mult * ATR    (price crashed far below KAMA)
Short: prune if price > KAMA + sigma_mult * ATR    (price pumped far above KAMA)
Target: Worst fill (highest entry for longs, lowest entry for shorts)
```

#### Method 2: Oldest Trade Pruning (Stale Capital)
```
prune if age > max_position_age_hours * 3600
Target: Oldest fill
```
Capital locked in old positions can't be redeployed. This is a time-based exit for fills that the grid failed to close through normal take-profit.

#### Method 3: Gap Pruning (Broken Grid)
```
prune if |price - fill_price| > gap_mult * grid_spacing
Target: Most distant fill
```
If price has moved so far from a fill that it's outside the grid's range, the fill is orphaned. Close it and let the grid regenerate at the new price level.

#### Method 4: Funding Cost Pruning (Bleeding)
```
prune if fill.funding_cost > cost_ratio * grid_profit_potential
Target: Fill with highest accumulated funding cost
```
Some fills sit open for so long that their accumulated funding costs exceed what a grid cycle would profit. Close them before they eat further into returns.

#### Method 5: Profit Offset Pruning (Subsidized Exit)
```
prune if accumulated_profit > worst_loss * 0.5
Target: Worst underwater fill
```
Use previously realized profit as a buffer to close losing positions at a reduced net cost. This cleans up the book while the strategy is still ahead.

---

## Main Loop: Step-by-Step Execution

**File**: `engine/strategy.py` | **Method**: `GridStrategyV4.run()`

For every bar from index 1 to N:

```
1. READ REGIME         Use indicators from bar i-1 (KAMA, ER, ATR, Z-Score)
                       ADX veto: if ADX < 25, all TREND signals overridden to NOISE
                       Current bar i provides only OHLC for fill matching

2. CIRCUIT BREAKER     If Z < -3.0: cancel all orders, set halted=True
                       If halted and Z > -1.0: set halted=False, regen grid
                       If halted: update equity, skip to next bar

3. STOP LOSS           Long: Stage 1 (candle CLOSE <= stop level, anti-wick):
                           Multi-fill: close 50%, widen stop to 1.5×
                       Stage 2 (wick Low <= stop level) or single fill: close 100%
                       Consecutive stop counter: ≥2 stops in 48 bars → halve order_pct
                       Short: symmetric

4. LIQUIDATION         If leveraged: check if price hit liquidation level
                       Close position at market if liquidated

5. PRUNING             For both long and short positions:
                       Run 5-method cycle, close targeted fill if triggered
                       + VaR pre-emptive: if drawdown > 75% of cap, close worst loser

6. TRAILING UP         If UPTREND/BREAKOUT_UP and ER > threshold:
                       If price > top grid level, shift anchor up 50%

7. VaR CHECK           Calculate total exposure VaR
                       If VaR > 15% of equity, block new orders

8. GRID GENERATION     If grid_needs_regen flag set AND VaR allows:
                         Cancel old orders
                         Compute Quarter-Kelly multiplier from last 50 regime-filtered trades
                         Apply stop-loss cooldown de-scaling if active
                         Apply weekend/low-vol de-scaling to max_position_pct
                         Calculate A-S reservation price (T=96)
                         Apply funding-aware sizing tilt in NOISE regime
                         Geometric grid levels with round-number avoidance
                         Place buy entries + sell TPs for long grid
                         Place sell entries + buy TPs for short grid

9. FILL MATCHING       Check each pending order against bar's High/Low
                       Process fills: update position, track PnL
                       Update Kelly trade history on PnL-generating fills

10. FUNDING RATE       Every 32 bars (8h): apply funding to all positions
                       Distribute cost to fills for funding pruning

11. LOG EQUITY         equity = wallet_balance + unrealized_pnl_long + unrealized_pnl_short
```

After all bars: close any remaining positions at the final close price.

---

## Backtesting: How We Avoid Lies

### Lookahead-Free Design

The most common backtesting error is lookahead bias: using future information to make current decisions. V4 enforces a strict separation:

```python
for i in range(1, n):
    # INDICATORS: computed from data up to bar i-1
    prev_kama = data['kama'][i - 1]
    prev_atr = data['atr'][i - 1]
    prev_regime = data['regime'][i - 1]
    prev_z = data['z_score'][i - 1]

    # PRICE ACTION: only current bar i (what we'd actually see)
    cur_high = data['high'][i]
    cur_low = data['low'][i]
    cur_close = data['close'][i]
```

**Verified by test**: `test_no_future_data_access` runs the strategy on two DataFrames that share the first 200 bars identically but differ afterward. If there's no lookahead, the first 200 bars of the equity curve must match exactly. This test passes.

### Conservative Fill Model

- Buy fills only when candle Low reaches order price (not Close)
- Sell fills only when candle High reaches order price (not Close)
- No assumption of execution at the opening price
- No partial fills

### Fee and Funding Simulation

- Maker fee: -0.005% (rebate)
- Taker fee: 0.02%
- Slippage: 0.05% per side (in config, applied through fees)
- Funding: Applied every 8 hours (32 bars) based on synthetic rate model
- All fees deducted from wallet balance immediately

---

## Optimizer: How Parameters Are Chosen

**File**: `optimizer.py`

The optimizer uses Optuna (Tree-Structured Parzen Estimator) with six anti-overfit measures:

### 1. Walk-Forward Validation
Data is split into overlapping windows. Each window: 70% train, 30% test. Parameters are evaluated only on the test portion.

### 2. Multi-Coin Averaging
The same parameters are tested across all coins (BTC, ETH, SOL). The score is the average across all coins and all windows. Parameters that only work on one asset are penalized.

### 3. Calmar Ratio Fitness
```
Fitness = Return / Max_Drawdown
```
This penalizes strategies that achieve high returns through high drawdowns.

### 4. Complexity Penalty
```
if trades > 500: penalty = -0.001 * (trades - 500)
```
Over-trading is a sign of over-optimization. A gentle penalty discourages it.

### 5. Stability Penalty
```
score -= 0.5 * std(scores_across_windows)
```
If a parameter set produces wildly different results across windows, it's likely overfit to specific market conditions.

### 6. Out-of-Sample Validation
After optimization, the best parameters are tested on a completely held-out 30% of data that was never used during optimization.

---

## Metrics: What We Measure

### Performance Metrics

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| Total Return % | `(final - initial) / initial * 100` | Raw profitability |
| Buy & Hold Return % | `(last_close - first_close) / first_close * 100` | Benchmark comparison |
| Max Drawdown % | `max((peak - trough) / peak) * 100` | Worst peak-to-trough loss |
| Sharpe Ratio | `mean(returns) / std(returns) * sqrt(96*365)` | Risk-adjusted return (annualized) |
| Sortino Ratio | `mean(returns) / std(negative_returns) * sqrt(96*365)` | Downside risk-adjusted return |
| Calmar Ratio | `total_return / max_drawdown` | Return per unit of drawdown |

### Trading Metrics

| Metric | What It Tells You |
|--------|-------------------|
| Win Rate % | Percentage of trades with positive PnL |
| Profit Factor | Gross profit / Gross loss (>1 = profitable) |
| Total Trades | Number of closed trades with non-zero PnL |
| Longs/Shorts Opened/Closed | Activity breakdown by direction |

### Risk Event Metrics

| Metric | What It Tells You |
|--------|-------------------|
| Stops (L/S) | How often ATR stops triggered per side |
| Prune Count | Total pruned fills (with type breakdown) |
| CB Triggers | How many crash halts occurred |
| Trailing Shifts | How many times grid shifted up in trends |
| VaR Blocks | How many times VaR blocked new orders |
| Liquidations | How many times positions were liquidated |
| Funding PnL | Net funding rate profit/loss |

**Annualization**: All ratio metrics use `sqrt(96 * 365)` as the annualization factor for 15-minute data (96 bars per day, 365 days per year).

---

## Parameter Reference

### Signal Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `kama_period` | 10 | 5-30 | ER lookback window (bars) |
| `kama_fast` | 2 | - | Fast EMA period for KAMA smoothing |
| `kama_slow` | 30 | - | Slow EMA period for KAMA smoothing |
| `atr_period` | 14 | - | ATR lookback window (bars) |
| `regime_threshold` | 0.15 | 0.05-0.3 | KAMA slope threshold for trend detection |
| `er_trend_threshold` | 0.5 | 0.3-0.7 | Minimum ER to classify as trend |

### Grid Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `grid_spacing_k` | 1.0 | 0.5-3.0 | Spacing multiplier on ATR |
| `grid_levels` | 10 | 3-20 | Grid levels per side |
| `order_pct` | 0.03 | 0.005-0.05 | Order size as fraction of capital |
| `spacing_floor` | 0.005 | - | Minimum spacing (0.5% of price) |
| `max_orders` | 500 | - | Maximum simultaneous orders |

### Inventory Parameters (Avellaneda-Stoikov)
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `gamma` | 0.5 | 0.1-2.0 | Risk aversion (higher = wider spreads) |
| `kappa` | 1.5 | 0.5-3.0 | Fill probability (higher = expect more fills) |
| `skew_factor` | 1.5 | - | Legacy skew multiplier |
| `max_inventory_per_side` | 10 | - | Max concurrent fills per direction |

### Trailing Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `trailing_enabled` | True | - | Enable/disable trailing up |
| `trailing_activation_er` | 0.65 | 0.5-0.8 | Minimum ER to activate trailing |
| `trailing_reserve_pct` | 0.25 | - | Capital reserved for buying higher |
| `trailing_cap_price` | None | - | Maximum price for grid anchor |

### Circuit Breaker Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `circuit_breaker_enabled` | True | - | Enable/disable circuit breaker |
| `halt_z_threshold` | -3.0 | -5 to -2 | Z-Score to trigger halt |
| `resume_z_threshold` | -1.0 | -2 to 0 | Z-Score to resume trading |

### Pruning Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `deviance_sigma` | 3.0 | 2.0-5.0 | ATR multiplier for deviance pruning |
| `max_position_age_hours` | 24 | 6-72 | Max fill age before pruning (hours) |
| `gap_prune_mult` | 3.0 | - | Grid spacing multiplier for gap pruning |
| `funding_cost_ratio` | 0.5 | - | Funding/profit threshold for pruning |

### Risk Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_drawdown_pct` | 0.15 | - | VaR hard cap (fraction of equity) |
| `atr_sl_mult` | 3.5 | 2.0-5.0 | Stop loss distance (ATR multiplier) |
| `max_position_pct` | 0.7 | - | Max notional per side vs equity |
| `leverage` | 1.0 | - | Position leverage (1.0 = spot equivalent) |

### Execution Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 10000 | Starting wallet balance ($) |
| `fee_maker` | -0.00005 | Maker fee rate (-0.005% rebate) |
| `fee_taker` | 0.0002 | Taker fee rate (0.02%) |
| `slippage` | 0.0005 | Slippage per side (0.05%) |
| `allow_short` | True | Enable short grid (hedge mode) |
| `funding_threshold` | 0.0003 | Adverse funding trigger (0.03%) |

---

## Testing

```bash
# Run all 57 tests
python3 -m pytest tests/ -v

# Run specific test module
python3 -m pytest tests/test_indicators.py -v
python3 -m pytest tests/test_strategy.py -v
python3 -m pytest tests/test_risk.py -v
python3 -m pytest tests/test_pruning.py -v
```

**Test coverage:**
- `test_indicators.py` (15 tests): ER range/warmup, KAMA tracking/smoothing, ATR positivity/volatility-scaling, Z-Score crash detection, regime detection (up/down/noise), rolling volatility
- `test_strategy.py` (12 tests): Valid result structure, equity start, no negative equity, metrics presence, hedge mode both-sides trading, long-only mode, circuit breaker trigger/disable, lookahead-free verification, trailing up, pruning count
- `test_risk.py` (16 tests): Margin tier lookup, maintenance margin, long/short/zero liquidation, VaR scaling/blocking/allowing, unrealized PnL (long/short profit/loss), funding PnL direction
- `test_pruning.py` (14 tests): Deviance trigger/no-trigger/worst-fill, oldest trigger/no-trigger, gap trigger/no-trigger, funding trigger/no-trigger, profit-offset trigger/no-trigger, priority ordering, empty position safety

---

## Realistic Expectations

**Where V4 excels:**
- Sideways/ranging markets (NOISE regime): This is grid trading's natural habitat. The strategy buys dips and sells rips systematically.
- Uptrends with trailing: Captures 80-90% of trend moves vs 20-30% for static grids.

**Where V4 mitigates but doesn't solve:**
- Downtrends: No long-only strategy profits in a monotonic downtrend. V4's Inventory Skew + Pruning + Circuit Breaker reduce drawdown from approximately -40% (static grid) to approximately -25%. The goal is capital preservation for trading the recovery.
- Black swans: Circuit Breaker halts trading and avoids "selling the bottom," but can't prevent drawdown from positions already held.

**What V4 cannot do:**
- Predict the future
- Guarantee profits in any market condition
- Eliminate drawdowns entirely

The strategy is designed for long-term edge through systematic risk management, not for winning every trade.

---

**Version**: V4.0 | **Date**: 2026-02-14 | **Status**: Complete (Backtest Mode)
