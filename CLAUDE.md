# GridStrategyV4 Architecture & Replication Guide

You are Claude. The user wants you to completely rebuild "GridStrategyV4", a highly advanced, cryptocurrency-native, hedge-mode grid trading system. This document contains the full specification, mathematical models, and architectural blueprints required to replicate the engine from scratch.

## Core Philosophy
This is not a toy grid bot. It is designed for 15-minute crypto perpetual futures. It uses an Unsupervised Machine Learning (GMM) or Kaufman Adaptive Moving Average (KAMA) regime filter to turn the grid on/off, an Avellaneda-Stoikov (A-S) market-making model to skew the grid away from toxic inventory, and a 5-method pruning system to surgically cut losing trades.

## Directory Structure
```text
core/
  kama.py       # ER, KAMA, and Regime FSM
  atr.py        # Wilder's ATR, Z-Score (circuit breaker), Rolling Volatility (for A-S model)
  adx.py        # Wilder's ADX (to veto weak trends)
  grid.py       # Geometric grid math, round-number avoidance
  inventory.py  # Avellaneda-Stoikov reservation price and optimal spread
  risk.py       # Binance tiered margin, iterative liquidation solver, VaR
  funding.py    # Synthetic funding rates, funding-aware sizing
  kelly.py      # Quarter-Kelly dynamically sized bets

engine/
  types.py      # PositionTracker (tracks independent fills), Trade labels
  matching.py   # Virtual OrderBook, Conservative Fill Model (Low<=P for buys)
  pruning.py    # 5-Method Gardener (Deviance, Oldest, Gap, Funding, Offset)
  strategy.py   # GridStrategyV4 (The 11-step orchestrator loop)

live/
  executor.py   # ccxt Binance wrapper (hedge mode, retries, rate limits)
  runner.py     # Live version of the 11-step loop parsing real-time 15m candles
  state.py      # Atomic JSON persistence for crash recovery
  logger.py     # PnL/Trade CSV and JSONL logging
  monitor.py    # Equity drawdown alerts and exchange sync
```

## Layer 1: Signal Math (core/)

1. **Efficiency Ratio (ER)**: `ER = abs(Close_t - Close_{t-n}) / Sum(abs(Close_i - Close_{i-1}))`
2. **KAMA**: 
   - `SC = (ER * (2/3 - 2/31) + 2/31)^2`
   - `KAMA = prev_KAMA + SC * (Close - prev_KAMA)`
3. **Regime FSM**: Normalized slope `(KAMA - prev_KAMA) / ATR`. 
   - State Machine: NOISE (0), UPTREND (1), DOWNTREND (-1), BREAKOUT_UP (2), BREAKOUT_DOWN (-2). 
   - *CRITICAL V4.2 ADDITION*: If ADX < 25, forcibly override UPTREND/DOWNTREND to NOISE (fakeout defense).
4. **Circuit Breaker (Z-Score)**: `Z = (Close - SMA_20) / ATR`. Halt trading if Z < -3.0. Resume if Z > -1.0.

## Layer 2: Grid & Inventory Math (core/)

1. **Avellaneda-Stoikov Model**:
   - `Normalized Inventory (q)`: `position_size / max_inventory` (clamped -1 to 1).
   - `Reservation Price (r)`: `r = mid_price - q * gamma * sigma^2 * TimeHorizon`
   - *Behavior*: If you hold Longs (q > 0), `r` drops below mid_price. You bid lower and ask lower to dump inventory.
2. **Geometric Grid**:
   - Spacing: `max(k * ATR, floor_pct * price)`.
   - Levels are exponential: `anchor * (1 + spacing_pct)^level`.
   - *Round-number defense*: Nudge levels 0.15% away from modulo-$1000/$100 price boundaries.
3. **Quarter-Kelly Sizing**:
   - `Kelly = (WinRate / AvgLoss) - ((1 - WinRate) / AvgWin)`
   - Bet size multiplier: `Kelly * 0.25` (computed over last 50 closed trades in matching regime).

## Layer 3: Execution & Engine (engine/ & live/)

1. **Hedge Mode**: Longs and Shorts are fundamentally distinct objects (`PositionTracker`). A short fill does NOT close a long fill.
2. **Fill Array**: A `PositionTracker` contains an array of `Fill` objects (qty, price, timestamp). We do not just track average entry. We track every single bullet fired so we can prune them individually.
3. **The 11-Step Loop (Executed every 15m candle)**:
   1. *Regime*: Read KAMA/ADX (from prev bar to prevent lookahead).
   2. *Circuit Breaker*: Check Z-Score. Cancel orders if halted.
   3. *Stop Loss*: 2-Stage ATR stop. Stage 1 (closes 50%) ONLY fires if the *candle closes* past the stop. Stage 2 (closes 100%) fires on the *wick*. Track consecutive stops; if >2 in 48 bars, halve order sizes (cooldown).
   4. *Liquidation Check*: Margin call logic.
   5. *Pruning*: Run the 5 methods. Also, if total unrealized drawdown > 75% of max VaR, force-close the worst single fill (Pre-emptive De-leverage).
   6. *Trailing Up*: If Uptrend and price > top grid, shift anchor up 50% of the gap.
   7. *VaR Check*: Stop new orders if `Portfolio * 1.65 * sigma > VaR_Limit`.
   8. *Grid Generation*: Center A-S model, Kelly-size the orders, tilt sizing toward funding (receive side +25%, pay side -25%).
   9. *Fill Matching*: Backtest (Low <= Price for buys) or Live (ccxt order polling).
   10. *Funding*: Subtract funding costs every 8 hours.
   11. *Log Equity*.

## Layer 4: Pruning Systems (engine/pruning.py)
Iterate through all open fills. First condition to trigger executes a market close for *that specific fill qty*.
1. **Deviance**: Price is > 3 ATR away from KAMA.
2. **Oldest**: Fill is > 24 hours old (stale capital).
3. **Gap**: Price is > 3 grid spacings away from the fill.
4. **Funding**: Accumulated funding cost for this fill > 50% of expected grid profit.
5. **Offset**: We have enough realized profit today to subsidize closing our worst underwater fill at breakeven net-equity.

## Live Trading Constraints
- **Atomic State**: When saving `/data/live_state/state.json`, write to a `.tmp` file and `os.rename` to prevent corruption if the bot crashes mid-write.
- **Min Notional**: Binance refuses orders < $5.0. Ensure the code dynamically calculates minimum quantity sizes based on the asset before sending orders.
- **REST Polling**: Do not use websockets. Poll `fetch_open_orders()` to deduce fills, then sync positions.

You are expected to write production-grade Python using `numpy` and `pandas`. Implement the formulas above strictly.
