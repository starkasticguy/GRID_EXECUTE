"""
GridStrategyV4 Full Configuration.

All parameters for the hedge-mode grid trading engine.
"""

STRATEGY_PARAMS = {
    # ─── Signal (KAMA/ER Regime Detection) ────────────────────
    'kama_period': 29,              # ER lookback (150 min on 15m)
    'kama_fast': 2,                 # Fast EMA period for KAMA
    'kama_slow': 30,                # Slow EMA period for KAMA
    'atr_period': 14,               # ATR lookback
    'regime_threshold': 0.2940875104844324,       # θ for KAMA slope FSM
    'er_trend_threshold': 0.6248712121898943,      # ER above this = trend regime

    # ─── Grid ─────────────────────────────────────────────────
    'grid_spacing_k': 0.5000844670481057,          # δ = k × ATR
    'grid_levels': 4,              # Levels per side
    'max_orders': 500,              # Max simultaneous orders
    'spacing_floor': 0.005,         # Min spacing (0.5% of price)
    'order_pct': 0.04999719568696284,              # Order size as % of capital

    # ─── Inventory (Avellaneda-Stoikov) ───────────────────────
    'gamma': 0.8073486673188088,                   # Risk aversion (0.1-2.0)
    'kappa': 2.1396980608779472,                   # Fill probability parameter
    'skew_factor': 1.5,             # Legacy skew multiplier
    'max_inventory_per_side': 10,   # Max fills per direction

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
    'max_position_age_hours': 9,   # Oldest trade pruning (hours)
    'deviance_sigma': 4.679956065163205,          # Deviance pruning (× ATR from KAMA)
    'gap_prune_mult': 3.0,          # Gap pruning (× grid spacing)
    'funding_cost_ratio': 0.5,      # Funding pruning threshold

    # ─── Risk Management ──────────────────────────────────────
    'max_drawdown_pct': 0.15,       # VaR hard cap (15% of equity)
    'var_confidence': 0.95,         # VaR confidence level
    'atr_sl_mult': 3.3285830660169395,             # Stop loss = entry ± mult × ATR
    'max_position_pct': 0.7,        # Max notional per side vs equity

    # ─── Funding Rate ─────────────────────────────────────────
    'funding_threshold': 0.0003,    # 0.03% adverse funding trigger

    # ─── Execution ────────────────────────────────────────────
    'initial_capital': 10000.0,
    'fee_maker': -0.00005,          # -0.005% maker rebate
    'fee_taker': 0.0002,            # 0.02% taker fee
    'slippage': 0.0005,             # 0.05% per side
    'leverage': 2.0,                # 1x default (spot-equivalent)
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

# ─── Optimizer Parameter Space ────────────────────────────────
# Used by optimizer.py for Bayesian/GA search
OPTIMIZER_SPACE = {
    'kama_period':      {'type': 'int',   'low': 5,    'high': 30},
    'grid_spacing_k':   {'type': 'float', 'low': 0.5,  'high': 3.0},
    'grid_levels':      {'type': 'int',   'low': 3,    'high': 20},
    'gamma':            {'type': 'float', 'low': 0.1,  'high': 2.0},
    'kappa':            {'type': 'float', 'low': 0.5,  'high': 3.0},
    'order_pct':        {'type': 'float', 'low': 0.005,'high': 0.05},
    'atr_sl_mult':      {'type': 'float', 'low': 2.0,  'high': 5.0},
    'regime_threshold': {'type': 'float', 'low': 0.05, 'high': 0.3},
    'er_trend_threshold':{'type': 'float', 'low': 0.3,  'high': 0.7},
    'trailing_enabled': {'type': 'cat',   'choices': [True, False]},
    'allow_short':      {'type': 'cat',   'choices': [True, False]},
    'max_position_age_hours': {'type': 'int', 'low': 6, 'high': 72},
    'deviance_sigma':   {'type': 'float', 'low': 2.0,  'high': 5.0},
}
