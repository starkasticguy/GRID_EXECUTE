"""
GridStrategyV4 Full Configuration.

All parameters for the hedge-mode grid trading engine.
"""

STRATEGY_PARAMS = {
    # ─── Signal (KAMA/ER + ADX Regime Detection) ────────────────
    'kama_period': 10,              # ER lookback (150 min on 15m)
    'kama_fast': 2,                 # Fast EMA period for KAMA
    'kama_slow': 30,                # Slow EMA period for KAMA
    'atr_period': 14,               # ATR lookback
    'adx_period': 14,               # ADX lookback (Wilder's standard)
    'adx_trend_threshold': 34.3383,    # ADX below this = no real trend (NOISE override)
    'regime_threshold': 0.0500612,       # θ for KAMA slope FSM (was 0.15 — less sensitive)
    'er_trend_threshold': 0.679668,     # ER above this = trend regime (was 0.5 — harder to enter trend)
    'regime_hysteresis_bars': 4,    # Bars to confirm regime change (prevents whipsaw)
    'regime_timeframe_mult': 4,     # HTF multiplier for regime (4 = 1H on 15m)
    'use_ml_regime': True,         # Use Gaussian Mixture Model for state classification instead of KAMA

    # ─── Grid ────────────────────────────────────────────────
    'grid_spacing_k': 1.27491,              # ETH-tuned: 1.6×ATR ≈ $24-26 spacing at $1500-3000. ETH's smoother candles fill TPs at this range
    'grid_levels': 8,               # Levels per side (was 2 — one extra level for more grid cycles)
    'max_orders': 500,              # Max simultaneous orders
    'spacing_floor': 0.00450328,         # Min spacing (0.6% of price, was 0.5%)
    'order_pct': 0.07,              # ETH-tuned: 7% per fill × 3 levels = 21% per side (up from 5%)
    'regen_drift_mult': 1.04053,        # ETH-tuned: tighter regen threshold (1.15× spacing). More grid regens → more TP fills
    'adaptive_floor_scale': 0.650928,    # Vol-scale for per-coin adaptive floor
    'tp_concentration': 0.867354,        # Fraction of TP qty at nearest level (rest spread evenly)
    'grid_mode': 'geometric',       # 'geometric' (pct-spaced, round-num avoidance) or 'arithmetic'

    # ─── Inventory (Avellaneda-Stoikov) ───────────────────────
    'gamma': 1.24686,                   # Risk aversion (was 1.2)
    'kappa': 1.83792,                   # Fill probability parameter
    'skew_factor': 0.8,             # Legacy skew multiplier
    'max_inventory_per_side': 4,    # Max fills per direction (was 3)
    'as_time_horizon': 192.919,        # A-S holding time horizon in bars (96 = 24h at 15m)

    # ─── Trailing Up ──────────────────────────────────────────
    'trailing_enabled': False,
    'trailing_activation_er': 0.65, # ER threshold for trailing (was 0.65 — harder to activate)
    'trailing_reserve_pct': 0.25,   # Reserve 25% capital for buying higher
    'trailing_cap_price': None,     # Max price to trail to (safety)

    # ─── Circuit Breaker ──────────────────────────────────────
    'circuit_breaker_enabled': True,
    'halt_z_threshold': -3.0,       # Z-score to trigger halt
    'resume_z_threshold': -1.0,     # Z-score to resume trading

    # ─── Pruning (5-Method Gardener) ──────────────────────────
    'max_position_age_hours': 24,   # Oldest trade pruning (was 39 — free locked capital faster)
    'deviance_sigma': 3.99983,              # Deviance pruning (was 3.32 — slightly tighter; 2.5 fired too often on SOL)
    'gap_prune_mult': 3.04177,              # Gap pruning (was 2.45 — tighter with narrower grid)
    'funding_cost_ratio': 0.400804,      # Funding pruning threshold
    'offset_prune_ratio': 3.44696,          # ETH-tuned: buffer must cover 400% of loss (very conservative, fewer bleeds)
    'prune_cooldown_bars': 20,      # ETH-tuned: 6h cooldown (was 8 = 2h). Holds fills longer → more TP fills

    # ─── Risk Management ──────────────────────────────────────
    'max_drawdown_pct': 0.163156,       # VaR hard cap (12% of equity)
    'var_confidence': 0.95,         # VaR confidence level
    'atr_sl_mult': 4.20043,                 # ETH-tuned: 3.5×ATR ≈ $49-56 from entry. ETH's calmer candles need tighter stops than SOL
    'max_position_pct': 0.583787,           # Was 0.376 — headroom for 5 grid levels (5% × 5 levels × 3× leverage = 75% notional, cap at 50%)

    # ─── Funding Rate ─────────────────────────────────────────
    'funding_threshold': 0.0003,    # 0.03% adverse funding trigger

    # ─── Execution ────────────────────────────────────────────
    'initial_capital': 300,
    'fee_maker': 0.0002,            # 0.02% maker fee (Binance actual)
    'fee_taker': 0.0005,            # 0.05% taker fee (Binance actual)
    'slippage': 0.0005,             # 0.05% per side
    'leverage': 4.0,                # 2x leverage (reduced from 3x — less stop-loss dollar impact)
    'allow_short': True,            # Enable short grid

    # ─── Risk De-scaling (new) ────────────────────────────────
    'stop_cooldown_bars': 82,       # Bars de-scale lasts after consecutive stops (12h on 15m)
    'stop_cooldown_thresh': 2,      # # of stops within window that triggers de-scale
    'low_volume_threshold': 0.53587,    # Volume < 50% of 7d SMA = low-liquidity period
    'funding_harvest_threshold': 0.00020853,  # Funding rate above this triggers sizing tilt
    'kelly_window': 37,             # Rolling trades window for Quarter-Kelly sizing
}

# ─── Live Trading Configuration ─────────────────────────────────
LIVE_CONFIG = {
    # Exchange Connection
    'exchange_id': 'binance',
    'market_type': 'future',         # USDM perpetual futures
    'position_mode': 'hedge',        # Hedge mode (dual side)
    'timeframe': '15m',

    # Candle Feed
    'feed_mode': 'poll',             # 'poll' — fetch candles periodically
    'poll_interval_seconds': 10,     # Check interval within bar wait
    'candle_close_grace_seconds': 5, # Wait after bar close for exchange to settle

    # Indicator Buffer
    'buffer_size': 200,              # Rolling candle buffer (covers SMA(96) + headroom)

    # Order Management
    'order_sync_interval_seconds': 5,
    'max_retry_attempts': 3,
    'retry_delay_seconds': 2,

    # State Persistence
    'state_dir': 'data/live_state',
    'log_dir': 'data/live_logs',
    'log_level': 'INFO',

    # Safety
    'dry_run': False,
    'max_capital_pct': 0.95,             # Never deploy more than 95% of wallet
    'emergency_stop_loss_pct': 0.25,     # If equity drops 25%, emergency halt
    'position_sync_tolerance_pct': 0.02, # 2% tolerance for position mismatch
}

BACKTEST_CONFIG = {
    'coins': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    'start_date': '2025-09-01',
    'end_date': None,               # None = current date
    'timeframe': '15m',
}

# ─── Backtest Simulation Settings ─────────────────────────────
BACKTEST_FILL_CONF = {
    'slippage_pct': 0.0005,        # 0.05% slippage on market orders
    'fill_probability': 1.0,       # ETH/BTC futures are deep — 100% realistic (was 0.9)
}

# ─── Optimizer Parameter Space ────────────────────────────────
# Used by optimizer.py for Bayesian/GA search
# Organized by category. Ranges are constrained to prevent degenerate configs.
OPTIMIZER_SPACE = {
    # ─── Grid Structure (core profitability) ──────────────────
    'grid_spacing_k':   {'type': 'float', 'low': 1.2,  'high': 2.0},   # ETH-tuned: 1.2-2.0×ATR (1.6 optimal; tighter hurts PF, wider kills fill rate)
    'spacing_floor':    {'type': 'float', 'low': 0.004, 'high': 0.010}, # Min spacing (% of price)
    'grid_levels':      {'type': 'int',   'low': 3,    'high': 8},      # Levels per side (min raised — 2 was too sparse)
    'regen_drift_mult': {'type': 'float', 'low': 1.0,  'high': 1.5},   # ETH-tuned: 1.1-1.5 optimal (above 1.5 kills TP frequency)
    'adaptive_floor_scale': {'type': 'float', 'low': 0.5, 'high': 3.0}, # Adaptive floor vol-scale
    'tp_concentration': {'type': 'float', 'low': 0.4, 'high': 0.9},    # TP concentration at nearest level

    # ─── Inventory Control (THE key risk param) ───────────────
    'gamma':            {'type': 'float', 'low': 0.3,  'high': 2.0},    # A-S risk aversion
    'kappa':            {'type': 'float', 'low': 0.5,  'high': 3.0},    # A-S fill probability
    'max_inventory_per_side': {'type': 'int', 'low': 2, 'high': 6},     # Max fills per direction
    'as_time_horizon':  {'type': 'float', 'low': 32.0, 'high': 288.0},  # Inventory decay (8h to 72h)

    # ─── Regime Detection ─────────────────────────────────────
    'regime_hysteresis_bars': {'type': 'int',   'low': 1,    'high': 6},     # Debounce window
    'adx_trend_threshold':    {'type': 'float', 'low': 20.0, 'high': 35.0}, # ADX veto: below = force NOISE
 #  'regime_timeframe_mult': {'type': 'int', 'low': 1, 'high': 8},          # HTF multiplier
 #  'regime_threshold': {'type': 'float', 'low': 0.10, 'high': 0.35},       # KAMA slope theta
 #  'er_trend_threshold':{'type': 'float', 'low': 0.4, 'high': 0.7},        # ER threshold for trend

    # ─── Risk Management ──────────────────────────────────────
    'atr_sl_mult':         {'type': 'float', 'low': 3.0,  'high': 5.0},   # ETH-tuned: 3.5 optimal; below 3.0 stops fires too often on ETH wicks
    'max_position_pct':    {'type': 'float', 'low': 0.3,  'high': 0.7},   # Max notional per side vs equity
    'stop_cooldown_bars':  {'type': 'int',   'low': 24,   'high': 96},    # Bars de-scale after stop-storm (6h–24h on 15m)
    'low_volume_threshold':{'type': 'float', 'low': 0.30, 'high': 0.70}, # Volume < X% of 7d avg = low liquidity
 #  'stop_cooldown_thresh':{'type': 'int',   'low': 2,    'high': 4},     # Stops to trigger de-scale (range too small; keep=2)

    # ─── Pruning ──────────────────────────────────────────────
    'deviance_sigma':   {'type': 'float', 'low': 2.5,  'high': 4.0},    # ETH-tuned: ETH is calmer, 3.0-3.5 is the sweet spot
    'gap_prune_mult':   {'type': 'float', 'low': 1.5,  'high': 4.0},    # Gap pruning threshold (tightened — aligns with narrower grid)
    'offset_prune_ratio': {'type': 'float', 'low': 3.0, 'high': 6.0},   # ETH-tuned: 4.0 optimal (fewer bleeds vs 2.0)
    'prune_cooldown_bars': {'type': 'int',  'low': 16,  'high': 32},     # ETH-tuned: 24 bars = 6h optimal (short cooldown = rapid-fire bleed)

    # ─── Mode Switches ────────────────────────────────────────
    'kelly_window':              {'type': 'int',   'low': 30,     'high': 100},    # Rolling trade window for Quarter-Kelly
    'funding_harvest_threshold': {'type': 'float', 'low': 0.0001, 'high': 0.0005}, # Funding rate tilt trigger
 #  'trailing_enabled': {'type': 'cat',   'choices': [True, False]},
 #  'allow_short':      {'type': 'cat',   'choices': [True, False]},
}
