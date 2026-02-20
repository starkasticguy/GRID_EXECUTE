"""
GridStrategyV4 Full Configuration.

All parameters for the hedge-mode grid trading engine.
"""

STRATEGY_PARAMS = {
    # ─── Signal (KAMA/ER Regime Detection) ────────────────────
    'kama_period': 10,              # ER lookback (150 min on 15m)
    'kama_fast': 2,                 # Fast EMA period for KAMA
    'kama_slow': 30,                # Slow EMA period for KAMA
    'atr_period': 14,               # ATR lookback
    'regime_threshold': 0.15,       # θ for KAMA slope FSM
    'er_trend_threshold': 0.5,      # ER above this = trend regime

    # ─── Grid ─────────────────────────────────────────────────
    'grid_spacing_k': 0.4,          # δ = k × ATR (wider = more profit per fill)
    'grid_levels': 3,               # Levels per side (less exposure per cycle)
    'max_orders': 500,              # Max simultaneous orders
    'spacing_floor': 0.005,         # Min spacing (0.3% of price)
    'order_pct': 0.04,              # Order size as % of capital

    # ─── Inventory (Avellaneda-Stoikov) ───────────────────────
    'gamma': 1.2,                   # Risk aversion (0.1-2.0) — stronger inventory rejection
    'kappa': 1.5,                   # Fill probability parameter
    'skew_factor': 1.5,             # Legacy skew multiplier
    'max_inventory_per_side': 3,    # Max fills per direction (strict cap)

    # ─── Trailing Up ──────────────────────────────────────────
    'trailing_enabled': True,
    'trailing_activation_er': 0.65, # ER threshold for trailing
    'trailing_reserve_pct': 0.25,   # Reserve 25% capital for buying higher
    'trailing_cap_price': None,     # Max price to trail to (safety)

    # ─── Circuit Breaker ──────────────────────────────────────
    'circuit_breaker_enabled': True,
    'halt_z_threshold': -3.0,       # Z-score to trigger halt
    'resume_z_threshold': -1.0,     # Z-score to resume trading

    # ─── Pruning (5-Method Gardener) ──────────────────────────
    'max_position_age_hours': 12,   # Oldest trade pruning (hours) — give fills time to work
    'deviance_sigma': 2.5,          # Deviance pruning (× ATR from KAMA)
    'gap_prune_mult': 2.5,          # Gap pruning (× grid spacing)
    'funding_cost_ratio': 0.5,      # Funding pruning threshold

    # ─── Risk Management ──────────────────────────────────────
    'max_drawdown_pct': 0.12,       # VaR hard cap (12% of equity)
    'var_confidence': 0.95,         # VaR confidence level
    'atr_sl_mult': 3.0,            # Stop loss = entry ± mult × ATR (tighter = smaller losses)
    'max_position_pct': 0.5,        # Max notional per side vs equity

    # ─── Funding Rate ─────────────────────────────────────────
    'funding_threshold': 0.0003,    # 0.03% adverse funding trigger

    # ─── Execution ────────────────────────────────────────────
    'initial_capital': 300,
    'fee_maker': 0.0002,            # 0.02% maker fee (Binance actual)
    'fee_taker': 0.0005,            # 0.05% taker fee (Binance actual)
    'slippage': 0.0005,             # 0.05% per side
    'leverage': 3.0,                # 3x leverage (more margin safety)
    'allow_short': True,            # Enable short grid
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
    'start_date': '2024-01-01',
    'end_date': None,               # None = current date
    'timeframe': '15m',
}

# ─── Backtest Simulation Settings ─────────────────────────────
BACKTEST_FILL_CONF = {
    'slippage_pct': 0.0005,        # 0.05% slippage on market orders
    'fill_probability': 0.9,       # 90% chance of limit fill if touched
}

# ─── Optimizer Parameter Space ────────────────────────────────
# Used by optimizer.py for Bayesian/GA search
# Organized by category. Ranges are constrained to prevent degenerate configs.
OPTIMIZER_SPACE = {
    # ─── Grid Structure (core profitability) ──────────────────
    'grid_spacing_k':   {'type': 'float', 'low': 0.3,  'high': 2.0},   # ATR multiplier for spacing
 #   'spacing_floor':    {'type': 'float', 'low': 0.003, 'high': 0.009}, # Min spacing (% of price)
    'grid_levels':      {'type': 'int',   'low': 2,    'high': 8},      # Levels per side (was 3-20, too wide)
 #   'order_pct':        {'type': 'float', 'low': 0.01, 'high': 0.06},   # Order size as % of capital

    # ─── Inventory Control (THE key risk param) ───────────────
    'gamma':            {'type': 'float', 'low': 0.3,  'high': 2.0},    # A-S risk aversion
    'kappa':            {'type': 'float', 'low': 0.5,  'high': 3.0},    # A-S fill probability
    'max_inventory_per_side': {'type': 'int', 'low': 2, 'high': 6},     # Max fills per direction (CRITICAL)

    # ─── Regime Detection ─────────────────────────────────────
 #   'regime_threshold': {'type': 'float', 'low': 0.05, 'high': 0.3},    # KAMA slope theta
 #   'er_trend_threshold':{'type': 'float', 'low': 0.3, 'high': 0.7},    # ER threshold for trend

    # ─── Risk Management ──────────────────────────────────────
    'atr_sl_mult':      {'type': 'float', 'low': 2.0,  'high': 5.0},    # Stop loss ATR multiplier
    'max_position_pct': {'type': 'float', 'low': 0.3,  'high': 0.7},    # Max notional per side vs equity
#   'max_drawdown_pct': {'type': 'float', 'low': 0.08, 'high': 0.20},   # VaR hard cap

    # ─── Pruning ──────────────────────────────────────────────
 #   'max_position_age_hours': {'type': 'int', 'low': 6, 'high': 48},    # Max fill age (was 6-72, too wide)
 #   'deviance_sigma':   {'type': 'float', 'low': 2.0,  'high': 5.0},    # Deviance pruning threshold
#    'gap_prune_mult':   {'type': 'float', 'low': 2.0,  'high': 5.0},    # Gap pruning threshold
 #   'funding_cost_ratio':{'type': 'float', 'low': 0.3,  'high': 0.8},   # Funding pruning threshold

    # ─── Mode Switches ────────────────────────────────────────
    'trailing_enabled': {'type': 'cat',   'choices': [True, False]},
    'allow_short':      {'type': 'cat',   'choices': [True, False]},
}
