"""
LiveRunner — Main live execution orchestrator.

Replicates the 11-step backtest loop (engine/strategy.py) but operates
on real-time 15m candle data and synchronizes with Binance exchange state.

Key differences from backtest:
  - Indicators computed on a rolling buffer (not full dataset)
  - Fills detected via exchange order polling (not virtual OrderBook)
  - Orders placed via BinanceExecutor (not virtual OrderBook)
  - Real funding rates (not synthetic)
  - State persisted to disk after every bar
"""
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from core.kama import (
    calculate_er, calculate_kama, detect_regime,
    resample_ohlcv, apply_regime_hysteresis,
    REGIME_NOISE, REGIME_UPTREND, REGIME_DOWNTREND,
    REGIME_BREAKOUT_UP, REGIME_BREAKOUT_DOWN, REGIME_NAMES,
)
from core.atr import calculate_atr, calculate_z_score, calculate_rolling_volatility
from core.adx import calculate_adx
from core.grid import (
    generate_grid_levels, generate_geometric_grid_levels,
    calculate_order_qty, calculate_dynamic_spacing,
    calculate_adaptive_floor,
)
from core.inventory import get_skewed_grid_params, normalize_inventory
from core.risk import check_var_constraint, calculate_liquidation_price
from core.funding import apply_funding, should_bias_for_funding, is_funding_interval
from core.kelly import compute_kelly_fraction
from engine.types import (
    PositionTracker, SIDE_BUY, SIDE_SELL, DIR_LONG, DIR_SHORT,
    LABEL_BUY_OPEN_LONG, LABEL_SELL_CLOSE_LONG,
    LABEL_SELL_OPEN_SHORT, LABEL_BUY_CLOSE_SHORT,
    LABEL_STOP_LONG, LABEL_STOP_SHORT,
    LABEL_CIRCUIT_BREAKER, LABEL_LIQUIDATION,
    LABEL_PRUNE_VAR_WARNING,
)
from core.ml_regime import classify_regimes_gmm
from engine.pruning import run_pruning_cycle
from live.executor import BinanceExecutor
from live.state import StateManager
from live.logger import TradeLogger
from live.monitor import HealthMonitor
from live.telegram_notifier import (
    TelegramNotifier, fmt_start, fmt_stop, fmt_trade,
    fmt_bar_summary, fmt_equity_warning,
)

logger = logging.getLogger('live_runner')


class LiveRunner:

    def __init__(self, executor: BinanceExecutor, symbol: str,
                 strategy_config: dict, live_config: dict,
                 state_manager: StateManager, trade_logger: TradeLogger,
                 health_monitor: HealthMonitor,
                 telegram: TelegramNotifier | None = None):
        self.executor = executor
        self.symbol = symbol
        self.config = strategy_config
        self.live_config = live_config
        self.state = state_manager
        self.trade_logger = trade_logger
        self.monitor = health_monitor
        self.tg = telegram  # TelegramNotifier (None = disabled)

        # Core state (mirrors backtest GridStrategyV4)
        self.initial_capital = strategy_config['initial_capital']
        self.wallet_balance = 0.0
        self.pos_long = PositionTracker(side=1)
        self.pos_short = PositionTracker(side=-1)
        self.halted = False
        self.grid_needs_regen = True
        self.grid_anchor_long = 0.0
        self.grid_anchor_short = 0.0
        self.trailing_anchor = 0.0
        self._last_grid_spacing = 0.0
        self._prev_regime = 0

        # Partial stop loss state: tracks if first-stage stop already fired
        self._partial_stop_fired_long = False
        self._partial_stop_fired_short = False

        # Accumulated profit for offset pruning
        self.accumulated_profit_long = 0.0
        self.accumulated_profit_short = 0.0

        # ─── V4.2 State: Kelly, stop cooldown, volume ─────────────
        # Quarter-Kelly position sizing — rolling window of closed trade pnls
        _KELLY_WINDOW = strategy_config.get('kelly_window', 30)
        self._KELLY_WINDOW = _KELLY_WINDOW
        self._kelly_trades: list = []   # [{pnl, regime}, ...]

        # Stop-loss de-scaling: after ≥ N consecutive stops reduce size for K bars
        self._descale_until_bar = 0     # bar index until which to de-scale
        self._consecutive_stops = 0

        # Rolling volume SMA for weekend/low-volume de-scaling
        self._vol_history: list = []    # recent bar volumes (float)
        self._vol_sma7: float = 0.0     # 7-day (672-bar) running volume SMA
        self._VOL_WINDOW = 96 * 7       # 672 bars = 7 days of 15m candles

        # Metrics (same schema as backtest)
        self.metrics = {
            'longs_opened': 0, 'longs_closed': 0,
            'shorts_opened': 0, 'shorts_closed': 0,
            'stops_long': 0, 'stops_short': 0,
            'prune_count': 0, 'prune_types': {},
            'circuit_breaker_triggers': 0, 'trailing_shifts': 0,
            'funding_pnl': 0.0, 'var_blocks': 0, 'liquidations': 0,
        }

        # Rolling candle buffer
        self.candle_buffer = None  # pd.DataFrame
        self.buffer_size = live_config.get('buffer_size', 200)

        # Exchange order tracking: exchange_order_id -> internal info
        self.active_orders = {}
        # Conditional orders (TP/SL) tracked separately — they don't appear
        # in fetch_open_orders() until triggered, so _sync_exchange_state()
        # must not treat them as disappeared/filled.
        self.conditional_orders = {}

        # Bar counter and timing
        self.bar_count = 0
        self.last_processed_bar_ts = 0
        self._last_regime = 0  # Cached regime from last indicator computation

        # Shutdown flag
        self._running = False
        self._session_start = None

    # ─── Initialization ──────────────────────────────────────────

    def initialize(self, resume: bool = False) -> bool:
        """Initialize runner state from exchange + persisted state."""
        try:
            # 1. Fetch wallet balance
            balance = self.executor.get_balance()
            self.wallet_balance = balance['total']
            if self.wallet_balance < 1:
                logger.error(f"Wallet balance too low: ${self.wallet_balance:.2f}")
                return False
            logger.info(f"Wallet balance: ${self.wallet_balance:,.2f}")

            # 2. Try to restore from saved state
            if resume:
                saved = self.state.load()
                if saved:
                    self._restore_state(saved)
                    logger.info("State restored from disk")
                else:
                    logger.info("No saved state found, starting fresh")

            # 3. Warm up candle buffer
            self._warm_up_candle_buffer()
            if self.candle_buffer is None or len(self.candle_buffer) < 30:
                logger.error("Insufficient candle data for indicator warm-up")
                return False
            logger.info(f"Candle buffer warmed: {len(self.candle_buffer)} bars")

            # 4. Set initial anchors from current price
            last_close = self.candle_buffer['close'].iloc[-1]
            if self.grid_anchor_long == 0:
                self.grid_anchor_long = last_close
            if self.grid_anchor_short == 0:
                self.grid_anchor_short = last_close
            if self.trailing_anchor == 0:
                self.trailing_anchor = last_close

            # 5. Sync with exchange positions
            self._sync_positions_from_exchange()

            self._session_start = datetime.now(timezone.utc)
            logger.info("LiveRunner initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def _warm_up_candle_buffer(self):
        """Fetch historical candles for indicator warm-up."""
        timeframe = self.live_config.get('timeframe', '15m')
        candles = self.executor.get_latest_candles(
            self.symbol, timeframe, limit=self.buffer_size)

        if not candles:
            logger.error("Failed to fetch historical candles")
            return

        # Drop the last candle if it's still open (incomplete)
        # The last candle from fetch_ohlcv is usually the current (open) bar
        if len(candles) > 1:
            candles = candles[:-1]

        df = pd.DataFrame(candles,
                          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.astype({'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
                        'low': 'float64', 'close': 'float64', 'volume': 'float64'})
        self.candle_buffer = df
        if len(df) > 0:
            self.last_processed_bar_ts = int(df['timestamp'].iloc[-1])

    def _sync_positions_from_exchange(self):
        """Sync internal position trackers with exchange reality."""
        positions = self.executor.get_positions(self.symbol)

        ex_long = positions['long']
        ex_short = positions['short']

        # If we have no fills tracked but exchange shows a position,
        # create a synthetic fill to match
        if ex_long['size'] > 1e-8 and not self.pos_long.is_open:
            logger.info(f"Syncing exchange LONG position: "
                        f"{ex_long['size']} @ {ex_long['avg_entry']}")
            self.pos_long.add_fill(
                ex_long['avg_entry'], ex_long['size'],
                time.time() * 1000, 0)

        if ex_short['size'] > 1e-8 and not self.pos_short.is_open:
            logger.info(f"Syncing exchange SHORT position: "
                        f"{ex_short['size']} @ {ex_short['avg_entry']}")
            self.pos_short.add_fill(
                ex_short['avg_entry'], ex_short['size'],
                time.time() * 1000, 0)

    def _restore_state(self, saved: dict):
        """Restore runner state from a saved state dict."""
        self.wallet_balance = saved.get('wallet_balance', self.wallet_balance)
        self.halted = saved.get('halted', False)
        self.grid_anchor_long = saved.get('grid_anchor_long', 0.0)
        self.grid_anchor_short = saved.get('grid_anchor_short', 0.0)
        self.trailing_anchor = saved.get('trailing_anchor', 0.0)
        self.grid_needs_regen = True  # Always regenerate grid on restart
        self.accumulated_profit_long = saved.get('accumulated_profit_long', 0.0)
        self.accumulated_profit_short = saved.get('accumulated_profit_short', 0.0)
        self.bar_count = saved.get('bar_count', 0)
        self.last_processed_bar_ts = saved.get('last_processed_bar_ts', 0)
        self.active_orders = saved.get('active_orders', {})
        self.conditional_orders = saved.get('conditional_orders', {})
        self.metrics = saved.get('metrics', self.metrics)
        
        # Restore V4.2 components
        self._kelly_trades = saved.get('_kelly_trades', [])
        self._consecutive_stops = saved.get('_consecutive_stops', 0)
        self._descale_until_bar = saved.get('_descale_until_bar', 0)
        self._vol_history = saved.get('_vol_history', [])
        self._partial_stop_fired_long = saved.get('_partial_stop_fired_long', False)
        self._partial_stop_fired_short = saved.get('_partial_stop_fired_short', False)

        # Restore position fills
        for side_key, tracker in [('pos_long', self.pos_long),
                                   ('pos_short', self.pos_short)]:
            pos_data = saved.get(side_key, {})
            fills = pos_data.get('fills', [])
            for fill in fills:
                tracker.add_fill(
                    fill['price'], fill['qty'],
                    fill.get('timestamp', 0),
                    0)
                if fill.get('funding_cost', 0) > 0:
                    # Restore funding cost per fill
                    if tracker.fills:
                        tracker.fills[-1]['funding_cost'] = fill['funding_cost']

    # ─── Main Loop ───────────────────────────────────────────────

    def run(self):
        """Main execution loop. Runs until shutdown signal."""
        self._running = True
        logger.info(f"LiveRunner started for {self.symbol}")

        # Telegram: announce startup
        if self.tg:
            leverage = self.config.get('leverage', 1.0)
            mode = 'LIVE' if not self.live_config.get('dry_run') else 'DRY-RUN'
            self.tg.send_now(fmt_start(self.symbol, self.wallet_balance, leverage, mode))

        while self._running:
            try:
                # 1. Wait for new bar
                new_candle = self._wait_for_new_candle()
                if new_candle is None:
                    continue

                # 2. Append to buffer
                self._append_candle(new_candle)

                # 3. Recompute indicators
                indicators = self._compute_indicators()
                if indicators is None:
                    logger.warning("Indicator computation failed, skipping bar")
                    continue

                # 4. Sync with exchange (detect limit order fills)
                self._sync_exchange_state()

                # 4a. Check conditional orders (TP/SL) for fills between candles
                self._sync_conditional_orders()

                # 4b. Reconcile positions with exchange (catch anything missed)
                # Runs AFTER both sync methods so properly detected fills
                # aren't double-processed by the blunt reconciliation.
                try:
                    positions = self.executor.get_positions(self.symbol)
                    self._reconcile_positions(positions)
                except Exception as e:
                    logger.warning(f"Position reconciliation failed: {e}")

                # 4c. Refresh wallet balance from exchange
                # (picks up fills, funding, fees applied by Binance)
                try:
                    balance = self.executor.get_balance()
                    ex_balance = balance.get('total', self.wallet_balance)
                    if abs(ex_balance - self.wallet_balance) > 0.01:
                        logger.info(f"Wallet balance synced: "
                                    f"${self.wallet_balance:.2f} → ${ex_balance:.2f}")
                        self.wallet_balance = ex_balance
                except Exception as e:
                    logger.warning(f"Balance refresh failed: {e}")

                # 5. Execute strategy step
                self._execute_strategy_step(indicators)

                # 6. Save state
                self.state.save(self._serialize_state())
                self.state.cleanup_old_snapshots()

                # 7. Health check
                self.monitor.heartbeat()
                self.monitor.check_position_sync(
                    self.executor, self.symbol,
                    self.pos_long, self.pos_short)

                # Equity check
                cur_price = indicators['close'][-1]
                total_equity = self._total_equity(cur_price)
                eq_status = self.monitor.check_equity(total_equity)
                if eq_status == 'critical':
                    logger.critical("Emergency equity threshold breached!")
                    if self.tg:
                        self.tg.send(
                            f'💀 <b>CRITICAL EQUITY BREACH</b> — emergency shutdown\n'
                            f'Symbol: <code>{self.symbol}</code>\n'
                            f'Equity: <code>${total_equity:,.2f}</code>')
                    self.emergency_shutdown()
                    break
                elif eq_status == 'warning' and self.tg:
                    peak = self.monitor.peak_equity
                    dd = (peak - total_equity) / peak * 100 if peak > 0 else 0
                    self.tg.send(fmt_equity_warning(self.symbol, total_equity, peak, dd))

                # Telegram: periodic bar heartbeat every 4 bars (silent, no notification sound)
                if self.tg and self.bar_count % 4 == 0:
                    regime_str = {
                        0: 'NOISE', 1: 'UPTREND', -1: 'DOWNTREND',
                        2: 'BREAKOUT_UP', -2: 'BREAKOUT_DOWN',
                    }.get(self._last_regime, str(self._last_regime))
                    long_s = (f"{self.pos_long.size:.4f}@{self.pos_long.avg_entry:.2f}"
                              if self.pos_long.is_open else 'flat')
                    short_s = (f"{self.pos_short.size:.4f}@{self.pos_short.avg_entry:.2f}"
                               if self.pos_short.is_open else 'flat')
                    self.tg.send(
                        fmt_bar_summary(self.symbol, cur_price, regime_str,
                                        total_equity, long_s, short_s,
                                        self.grid_needs_regen),
                        silent=True)  # silent = no ding

                self.bar_count += 1

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — shutting down")
                self.shutdown()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                should_stop = self.monitor.report_error(e)
                if should_stop:
                    logger.critical("Too many consecutive errors — shutting down")
                    self.shutdown()
                    break
                time.sleep(30)

    def _wait_for_new_candle(self):
        """Wait for the next completed 15m candle."""
        timeframe_ms = 15 * 60 * 1000  # 15 minutes in ms
        grace_s = self.live_config.get('candle_close_grace_seconds', 5)
        poll_s = self.live_config.get('poll_interval_seconds', 10)

        # Calculate when the next bar closes
        now_ms = int(time.time() * 1000)
        next_bar_close = ((now_ms // timeframe_ms) + 1) * timeframe_ms

        # Sleep until bar close + grace
        wait_s = (next_bar_close - now_ms) / 1000.0 + grace_s
        if wait_s > 0:
            logger.info(f"Waiting {wait_s:.0f}s for next 15m bar close...")
            # Sleep in chunks so we can respond to shutdown and run mid-candle checks
            intracandle_interval = self.live_config.get('intracandle_check_seconds', 300)  # 5 min
            end_time = time.time() + wait_s
            last_check_t = time.time()
            while time.time() < end_time and self._running:
                time.sleep(max(0, min(poll_s, end_time - time.time())))
                # Fire mid-candle check every 5 min (but not right before bar close)
                now = time.time()
                if now - last_check_t >= intracandle_interval and (end_time - now) > poll_s:
                    try:
                        self._intracandle_check()
                    except Exception as e:
                        logger.warning(f"Intracandle check error: {e}")
                    last_check_t = now

        if not self._running:
            return None

        # Fetch latest candles and pick the completed one
        timeframe = self.live_config.get('timeframe', '15m')
        candles = self.executor.get_latest_candles(
            self.symbol, timeframe, limit=3)

        if not candles or len(candles) < 2:
            logger.warning("Failed to fetch candles")
            return None

        # Second-to-last is the most recently completed bar
        completed = candles[-2]
        candle_ts = int(completed[0])

        # Verify it's newer than last processed
        if candle_ts <= self.last_processed_bar_ts:
            logger.debug("No new completed bar yet")
            return None

        self.last_processed_bar_ts = candle_ts
        return {
            'timestamp': candle_ts,
            'open': float(completed[1]),
            'high': float(completed[2]),
            'low': float(completed[3]),
            'close': float(completed[4]),
            'volume': float(completed[5]),
        }

    def _append_candle(self, candle: dict):
        """Append a new candle to the rolling buffer."""
        new_row = pd.DataFrame([candle])
        self.candle_buffer = pd.concat(
            [self.candle_buffer, new_row], ignore_index=True)

        # Trim to buffer size
        if len(self.candle_buffer) > self.buffer_size:
            self.candle_buffer = self.candle_buffer.iloc[-self.buffer_size:].reset_index(drop=True)

    def _intracandle_check(self):
        """Mid-candle lightweight check: fill sync + stop loss only.

        Runs every 5 minutes during the inter-bar wait period.
        Does NOT recompute indicators or regenerate the grid.
        Provides faster stop loss reaction on intrabar crash moves.
        """
        # 1. Sync fills from exchange so positions are up to date
        try:
            self._sync_exchange_state()
            self._sync_conditional_orders()
        except Exception as e:
            logger.debug(f"Intracandle fill sync error: {e}")

        # Skip stop checks if no open positions
        if not self.pos_long.is_open and not self.pos_short.is_open:
            return

        # 2. Get current live price
        try:
            ticker = self.executor.get_ticker(self.symbol)
            cur_price = float(ticker.get('last') or ticker.get('close') or 0)
            if cur_price <= 0:
                return
        except Exception as e:
            logger.debug(f"Intracandle ticker fetch failed: {e}")
            return

        # 3. Get last known ATR from candle buffer
        if self.candle_buffer is None or len(self.candle_buffer) < 20:
            return
        closes = self.candle_buffer['close'].values.astype(np.float64)
        highs = self.candle_buffer['high'].values.astype(np.float64)
        lows = self.candle_buffer['low'].values.astype(np.float64)
        atr_arr = calculate_atr(highs, lows, closes, self.config['atr_period'])
        cur_atr = float(atr_arr[-1])
        if cur_atr <= 0:
            return

        atr_sl_mult = self.config.get('atr_sl_mult', 3.0)
        fee_rate = self.config.get('fee_taker', 0.0005)

        # 4. Check long stop (Emergency Stage 2 Only)
        if self.pos_long.is_open:
            # Mid-candle checks only fire on the extreme 1.5x full-stop wick threshold
            emergency_stop_price = self.pos_long.avg_entry - (atr_sl_mult * 1.5) * cur_atr
            if cur_price <= emergency_stop_price:
                logger.info(f"[INTRACANDLE] EMERGENCY STOP LOSS LONG: price={cur_price:.2f} <= stop={emergency_stop_price:.2f}")
                order = self.executor.place_market_order(
                    self.symbol, 'sell', self.pos_long.size, 'LONG', reduce_only=True)
                if order:
                    actual = float(order.get('average', cur_price) or cur_price)
                    fee = self.pos_long.size * actual * fee_rate
                    pnl = self.pos_long.close_all(actual, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_long'] += 1
                    self._partial_stop_fired_long = False
                    self.grid_needs_regen = True
                    self._log_trade_event(
                        int(time.time() * 1000), actual, 0,
                        LABEL_STOP_LONG, self._last_regime, pnl)

        if not self.pos_long.is_open:
            self._partial_stop_fired_long = False

        # 5. Check short stop (Emergency Stage 2 Only)
        if self.pos_short.is_open:
            # Mid-candle checks only fire on the extreme 1.5x full-stop wick threshold
            emergency_stop_price = self.pos_short.avg_entry + (atr_sl_mult * 1.5) * cur_atr
            if cur_price >= emergency_stop_price:
                logger.info(f"[INTRACANDLE] EMERGENCY STOP LOSS SHORT: price={cur_price:.2f} >= stop={emergency_stop_price:.2f}")
                order = self.executor.place_market_order(
                    self.symbol, 'buy', self.pos_short.size, 'SHORT', reduce_only=True)
                if order:
                    actual = float(order.get('average', cur_price) or cur_price)
                    fee = self.pos_short.size * actual * fee_rate
                    pnl = self.pos_short.close_all(actual, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_short'] += 1
                    self._partial_stop_fired_short = False
                    self.grid_needs_regen = True
                    self._log_trade_event(
                        int(time.time() * 1000), actual, 0,
                        LABEL_STOP_SHORT, self._last_regime, pnl)

        if not self.pos_short.is_open:
            self._partial_stop_fired_short = False


    def _compute_indicators(self):
        """Compute all indicators on the rolling buffer."""
        df = self.candle_buffer
        if df is None or len(df) < 30:
            return None

        try:
            closes = df['close'].values.astype(np.float64)
            highs = df['high'].values.astype(np.float64)
            lows = df['low'].values.astype(np.float64)
            opens = df['open'].values.astype(np.float64)
            timestamps = df['timestamp'].values.astype(np.float64)
            volumes = df['volume'].values.astype(np.float64) \
                if 'volume' in df.columns else np.ones(len(closes))

            kama_p = self.config['kama_period']
            atr_p = self.config['atr_period']

            atr = calculate_atr(highs, lows, closes, atr_p)

            # ─── Multi-timeframe regime (HTF KAMA/ER on resampled data) ───
            htf_mult = self.config.get('regime_timeframe_mult', 4)
            htf_close, htf_high, htf_low = resample_ohlcv(
                closes, highs, lows, mult=htf_mult)
            htf_atr = calculate_atr(htf_high, htf_low, htf_close, atr_p)
            er = calculate_er(htf_close, kama_p)
            kama = calculate_kama(htf_close, er,
                                   self.config['kama_fast'],
                                   self.config['kama_slow'])

            # ─── ADX veto: computed on 15m for sensitivity ─
            adx_period = self.config.get('adx_period', 14)
            adx_threshold = self.config.get('adx_trend_threshold', 25.0)
            adx = calculate_adx(highs, lows, closes, adx_period)

            # ─── Regime Detection (ML vs KAMA) ─────────────────────────
            if self.config.get('use_ml_regime', False):
                raw_regime = classify_regimes_gmm(closes, highs, lows, volumes)
            else:
                raw_regime = detect_regime(kama, er, htf_atr,
                                           self.config['regime_threshold'],
                                           self.config['er_trend_threshold'])
            # Apply ADX veto: if trend/breakout regime but ADX is weak → NOISE
            veto_regime = raw_regime.copy()
            for idx in range(len(veto_regime)):
                if abs(veto_regime[idx]) in (1, 2) and adx[idx] < adx_threshold:
                    veto_regime[idx] = REGIME_NOISE

            # ─── Hysteresis debounce ───
            hysteresis_bars = self.config.get('regime_hysteresis_bars', 3)
            regime = apply_regime_hysteresis(veto_regime, min_bars=hysteresis_bars)
            z_score = calculate_z_score(closes, atr, 20)
            volatility = calculate_rolling_volatility(closes, atr_p)

            # ─── Update rolling volume SMA (for weekend de-scaling) ──
            if len(volumes) > 0:
                cur_vol = float(volumes[-1])
                self._vol_history.append(cur_vol)
                if len(self._vol_history) > self._VOL_WINDOW:
                    self._vol_history = self._vol_history[-self._VOL_WINDOW:]
                self._vol_sma7 = float(np.mean(self._vol_history)) \
                    if self._vol_history else 0.0
            else:
                cur_vol = 0.0

            return {
                'open': opens, 'high': highs, 'low': lows, 'close': closes,
                'timestamp': timestamps,
                'kama': kama, 'er': er, 'atr': atr, 'adx': adx,
                'regime': regime, 'z_score': z_score,
                'volatility': volatility,
                'cur_volume': cur_vol,
            }
        except Exception as e:
            logger.error(f"Indicator computation failed: {e}", exc_info=True)
            return None

    # ─── Exchange State Sync ─────────────────────────────────────

    def _sync_exchange_state(self):
        """
        Synchronize internal state with exchange.
        Detect fills by comparing tracked orders with exchange open orders.
        """
        if not self.active_orders:
            return

        # Fetch current open orders
        exchange_orders = self.executor.get_open_orders(self.symbol)
        exchange_ids = {o['id'] for o in exchange_orders}

        # Find orders that disappeared (filled or cancelled)
        disappeared = []
        for oid, info in list(self.active_orders.items()):
            if oid not in exchange_ids:
                disappeared.append((oid, info))

        if not disappeared:
            return

        # Fetch positions BEFORE processing to compare size changes
        positions_before = self.executor.get_positions(self.symbol)
        prev_long_size = positions_before['long']['size']
        prev_short_size = positions_before['short']['size']

        for oid, order_info in disappeared:
            # Try to determine if order was filled or cancelled
            try:
                order_status = self.executor.exchange.fetch_order(oid, self.symbol)
                status = order_status.get('status', 'unknown')
                if status in ('closed',):
                    # Order was filled — use actual filled qty (not requested)
                    # to handle partial fills correctly
                    fill_price = float(order_status.get('average', 0) or
                                       order_status.get('price', 0) or
                                       order_info['price'])
                    filled_qty = float(order_status.get('filled',
                                                         order_info['qty']))
                    if filled_qty > 1e-12:
                        fill_info = order_info.copy()
                        fill_info['qty'] = filled_qty
                        logger.info(f"Order {oid} FILLED @ {fill_price:.2f} "
                                    f"qty={filled_qty}")
                        self._process_live_fill(fill_info, fill_price)
                elif status in ('canceled', 'cancelled', 'expired', 'rejected'):
                    logger.info(f"Order {oid} was {status}, not a fill")
                else:
                    logger.warning(f"Order {oid} unknown status '{status}', skipping")
            except Exception as e:
                # If we can't check status, compare position sizes to infer
                logger.warning(f"Could not fetch order {oid} status: {e}. "
                               f"Inferring from position change.")
                # Conservative: skip and let reconciliation handle it
            del self.active_orders[oid]

    def _sync_conditional_orders(self):
        """Check conditional orders (TP/SL) for fills that occurred between candles.

        Unlike regular limit orders which appear in fetch_open_orders(),
        conditional orders live on a separate Binance API. We must poll
        each tracked conditional order individually via fetch_conditional_order()
        to detect fills.
        """
        if not self.conditional_orders:
            return

        filled_ids = []

        for oid, order_info in list(self.conditional_orders.items()):
            try:
                order_status = self.executor.fetch_conditional_order(oid, self.symbol)
                if order_status is None:
                    continue

                status = order_status.get('status', 'unknown')

                if status == 'closed':
                    # Conditional order was triggered and filled
                    # Use actual filled qty (not requested) for partial fills
                    fill_price = float(
                        order_status.get('average', 0) or
                        order_status.get('price', 0) or
                        order_info['price'])
                    filled_qty = float(order_status.get('filled',
                                                         order_info['qty']))
                    if filled_qty > 1e-12:
                        fill_info = order_info.copy()
                        fill_info['qty'] = filled_qty
                        logger.info(
                            f"Conditional order {oid} ({order_info.get('order_type', '?')}) "
                            f"FILLED @ {fill_price:.4f} qty={filled_qty}")
                        self._process_live_fill(fill_info, fill_price)
                    filled_ids.append(oid)

                elif status in ('canceled', 'cancelled', 'expired', 'rejected'):
                    logger.info(f"Conditional order {oid} was {status}, removing from tracking")
                    filled_ids.append(oid)

                # 'open' status → still waiting, leave in dict

            except Exception as e:
                logger.warning(f"Error checking conditional order {oid}: {e}")

        # Remove processed orders
        for oid in filled_ids:
            del self.conditional_orders[oid]

    def _process_live_fill(self, order_info: dict, fill_price: float):
        """Process a detected fill from exchange sync."""
        direction = order_info['direction']
        side = order_info['side']
        qty = order_info['qty']
        reduce_only = order_info['reduce_only']
        fee_rate = self.config['fee_taker']
        fee = qty * fill_price * fee_rate
        timestamp = time.time() * 1000

        # Use cached regime from last indicator computation
        regime = self._last_regime

        pnl = 0.0
        label = ''

        if direction == DIR_LONG:
            if side == 'buy' and not reduce_only:
                # Opening long
                self.pos_long.add_fill(fill_price, qty, timestamp, fee)
                self.wallet_balance -= fee
                self.metrics['longs_opened'] += 1
                label = LABEL_BUY_OPEN_LONG
            elif side == 'sell':
                # Closing long (take-profit)
                pnl = self.pos_long.close_fill(fill_price, qty, fee)
                self.wallet_balance += pnl
                self.metrics['longs_closed'] += 1
                label = LABEL_SELL_CLOSE_LONG
                if pnl > 0:
                    self.accumulated_profit_long += pnl
        elif direction == DIR_SHORT:
            if side == 'sell' and not reduce_only:
                # Opening short
                self.pos_short.add_fill(fill_price, qty, timestamp, fee)
                self.wallet_balance -= fee
                self.metrics['shorts_opened'] += 1
                label = LABEL_SELL_OPEN_SHORT
            elif side == 'buy':
                # Closing short (take-profit)
                pnl = self.pos_short.close_fill(fill_price, qty, fee)
                self.wallet_balance += pnl
                self.metrics['shorts_closed'] += 1
                label = LABEL_BUY_CLOSE_SHORT
                if pnl > 0:
                    self.accumulated_profit_short += pnl

        if label:
            trade = {
                'timestamp': int(timestamp),
                'datetime': datetime.now(timezone.utc).isoformat(),
                'symbol': self.symbol,
                'price': fill_price,
                'qty': qty,
                'label': label,
                'regime': regime,
                'pnl': pnl,
                'exchange_order_id': order_info.get('exchange_id', ''),
                'fill_price': fill_price,
                'fee': fee,
                'wallet_balance_after': self.wallet_balance,
                'equity_after': self._total_equity(fill_price),
                'pos_long_size': self.pos_long.size,
                'pos_short_size': self.pos_short.size,
            }
            self.trade_logger.log_trade(trade)
            self.state.save_trade(trade)

            # Update Kelly trade memory for dynamic sizing (#5)
            if pnl != 0:
                self._kelly_trades.append({
                    'pnl': pnl,
                    'regime': regime,
                })
                kelly_window = self.config.get('kelly_window', 50)
                if len(self._kelly_trades) > kelly_window:
                    self._kelly_trades = self._kelly_trades[-kelly_window:]

        # Always regenerate grid on any fill so that conditional orders
        # (TPs and SLs) are placed immediately for the new/changed position.
        # Without this, fills near the anchor would skip regen and leave
        # positions unprotected (no TP/SL on exchange).
        self.grid_needs_regen = True

    def _reconcile_positions(self, exchange_positions: dict):
        """Adjust internal tracker if exchange shows different position size."""
        tolerance = self.live_config.get('position_sync_tolerance_pct', 0.02)

        for side_key, tracker in [('long', self.pos_long), ('short', self.pos_short)]:
            ex_size = exchange_positions[side_key]['size']
            int_size = tracker.size

            if abs(ex_size - int_size) > max(ex_size, int_size, 1e-9) * tolerance:
                if abs(ex_size - int_size) > 1e-8:
                    logger.warning(
                        f"Reconciling {side_key.upper()}: "
                        f"exchange={ex_size:.8f} → internal was {int_size:.8f}")
                    # Exchange is truth — adjust internal to match
                    if ex_size > int_size:
                        # Position grew externally (manual open)
                        diff = ex_size - int_size
                        avg_entry = exchange_positions[side_key].get('avg_entry', 0)
                        if avg_entry > 0:
                            tracker.add_fill(avg_entry, diff, time.time() * 1000, 0)
                    elif ex_size < int_size and ex_size < 1e-8:
                        # Position was fully closed externally
                        tracker.size = 0.0
                        tracker.avg_entry = 0.0
                        tracker.fills.clear()
                        tracker.num_fills = 0
                    elif ex_size < int_size:
                        # Position was partially closed externally
                        excess = int_size - ex_size
                        avg_entry = exchange_positions[side_key].get('avg_entry', 0) or tracker.avg_entry
                        tracker.close_fill(avg_entry, excess, 0)

    # ─── Strategy Step (11-step loop) ────────────────────────────

    def _execute_strategy_step(self, indicators: dict):
        """Execute one iteration of the 11-step strategy loop."""
        # Need at least 2 bars for prev/current
        if len(indicators['close']) < 2:
            return

        # ─── PREV BAR indicators (no lookahead) ─────────────
        prev_kama = indicators['kama'][-2]
        prev_atr = indicators['atr'][-2]
        prev_regime = int(indicators['regime'][-2])
        prev_z = indicators['z_score'][-2]
        prev_er = indicators['er'][-2]
        prev_vol = indicators['volatility'][-2]

        # Cache regime for use in fill processing
        self._last_regime = prev_regime

        # ─── CURRENT BAR price action ────────────────────────
        cur_close = indicators['close'][-1]
        cur_high = indicators['high'][-1]
        cur_low = indicators['low'][-1]
        cur_time = indicators['timestamp'][-1]

        # Extract config params
        leverage = self.config.get('leverage', 1.0)
        allow_short = self.config.get('allow_short', True)
        fee_rate = self.config['fee_taker']
        order_pct = self.config['order_pct']
        grid_levels = self.config['grid_levels']
        gamma = self.config['gamma']
        kappa = self.config.get('kappa', 1.5)
        spacing_k = self.config['grid_spacing_k']
        spacing_floor = self.config.get('spacing_floor', 0.005)
        max_inv_per_side = self.config.get('max_inventory_per_side', 10)
        atr_sl_mult = self.config.get('atr_sl_mult', 3.5)
        max_position_pct = self.config.get('max_position_pct', 0.7)
        trailing_enabled = self.config.get('trailing_enabled', True)
        trailing_er = self.config.get('trailing_activation_er', 0.65)
        cb_enabled = self.config.get('circuit_breaker_enabled', True)
        halt_z = self.config.get('halt_z_threshold', -3.0)
        resume_z = self.config.get('resume_z_threshold', -1.0)
        max_dd_pct = self.config.get('max_drawdown_pct', 0.15)

        # ─── 1. STATUS HEADER ─────────────────────────────────
        regime_names = {0: 'NOISE', 1: 'UPTREND', -1: 'DOWNTREND',
                        2: 'BREAKOUT_UP', -2: 'BREAKOUT_DOWN'}
        regime_str = regime_names.get(prev_regime, str(prev_regime))
        long_str = (f"L:{self.pos_long.size:.4f}@{self.pos_long.avg_entry:.2f}"
                    if self.pos_long.is_open else "L:flat")
        short_str = (f"S:{self.pos_short.size:.4f}@{self.pos_short.avg_entry:.2f}"
                     if self.pos_short.is_open else "S:flat")
        equity = self._total_equity(cur_close)
        logger.info(
            f"\n{'─'*60}\n"
            f"  BAR | price={cur_close:.2f}  ATR={prev_atr:.2f}  Z={prev_z:.2f}  ER={prev_er:.2f}"
            f"  regime={regime_str}"
            f"  wallet=${self.wallet_balance:.2f}  equity=${equity:.2f}"
            f"\n  pos  | {long_str}  {short_str}\n{'─'*60}"
        )

        # ─── 2. CIRCUIT BREAKER ──────────────────────────────────
        if cb_enabled:
            if not self.halted and prev_z < halt_z:
                logger.info(f"  [CB] Z={prev_z:.2f} breached halt threshold {halt_z}. Cancelling all orders.")
                self.halted = True
                self.metrics['circuit_breaker_triggers'] += 1
                self.executor.cancel_all_orders(self.symbol)
                self.active_orders.clear()
                self.conditional_orders.clear()
                if self.tg:
                    self.tg.send(
                        f'⚡ <b>CIRCUIT BREAKER HALT</b>\n'
                        f'Symbol: <code>{self.symbol}</code>\n'
                        f'Z-Score: <code>{prev_z:.2f}</code> (threshold {halt_z})\n'
                        f'Price: <code>${cur_close:,.2f}</code>\n'
                        f'All orders cancelled. Holding positions.')
                self.trade_logger.log_event(
                    'CIRCUIT_BREAKER',
                    f'Z-Score: {prev_z:.2f} < {halt_z}. All orders cancelled.')
                self._log_trade_event(
                    cur_time, cur_close, 0, LABEL_CIRCUIT_BREAKER, prev_regime, 0)

            if self.halted and prev_z > resume_z:
                logger.info(f"  [CB] Z={prev_z:.2f} recovered above {resume_z}. Resuming trading.")
                self.halted = False
                self.grid_needs_regen = True
                if self.tg:
                    self.tg.send(
                        f'🟢 <b>CIRCUIT BREAKER RESUME</b>\n'
                        f'Symbol: <code>{self.symbol}</code>\n'
                        f'Z-Score recovered: <code>{prev_z:.2f}</code>\n'
                        f'Trading resumed.')
                self.trade_logger.log_event(
                    'CB_RESUME',
                    f'Z-Score: {prev_z:.2f} > {resume_z}. Trading resumed.')

        if self.halted:
            logger.info(f"  [CB] HALTED — Z={prev_z:.2f}. Skipping grid/stops until recovery.")
            self._update_equity(cur_close)
            self._log_equity_bar(cur_time, cur_close, prev_regime)
            return

        # ─── 3. STOP LOSS (ATR-based, partial) ──────────────────
        if self.pos_long.is_open:
            effective_mult = atr_sl_mult * 1.5 if self._partial_stop_fired_long else atr_sl_mult
            stop_price = self.pos_long.avg_entry - effective_mult * prev_atr
            distance_pct = (cur_low - stop_price) / stop_price * 100
            
            stage1_trigger = cur_close <= stop_price
            stage2_trigger = cur_low <= stop_price

            logger.info(f"  [SL]  LONG stop={stop_price:.2f}  low={cur_low:.2f} close={cur_close:.2f} "
                        f"distance={distance_pct:+.2f}%{'  ← STAGE2' if stage2_trigger else ''}{' ← STAGE1' if stage1_trigger else ''}")

            if stage1_trigger and self.pos_long.num_fills > 1 and not self._partial_stop_fired_long:
                # Stage 1: partial stop — close 50% on candle CLOSE breach (anti-wick)
                close_qty = self.pos_long.size * 0.5
                logger.info(f"PARTIAL STOP LONG (50%): close={cur_close:.2f} <= stop={stop_price:.2f}, "
                            f"closing {close_qty:.6f} of {self.pos_long.size:.6f}")
                order = self.executor.place_market_order(
                    self.symbol, 'sell', close_qty,
                    'LONG', reduce_only=True)
                if order is None:
                    logger.warning(f"Partial stop loss market order failed for LONG")
                else:
                    actual_price = float(order.get('average', stop_price) or stop_price)
                    fee = close_qty * actual_price * fee_rate
                    pnl = self.pos_long.close_fill(actual_price, close_qty, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_long'] += 1
                    self._partial_stop_fired_long = True
                    self.grid_needs_regen = True
                    # ─── Stop-cooldown counter (matches backtest) ─────────
                    self._consecutive_stops += 1
                    cooldown_thresh = self.config.get('stop_cooldown_thresh', 2)
                    cooldown_bars = self.config.get('stop_cooldown_bars', 48)
                    if self._consecutive_stops >= cooldown_thresh:
                        self._descale_until_bar = self.bar_count + cooldown_bars
                        logger.info(f"  [SL]  Consecutive stops={self._consecutive_stops} ≥ "
                                    f"{cooldown_thresh}: de-scaling for {cooldown_bars} bars")
                    self._log_trade_event(
                        cur_time, actual_price, close_qty, LABEL_STOP_LONG, prev_regime, pnl)
            elif stage2_trigger:
                # Stage 2 (or single fill): close everything
                logger.info(f"STOP LOSS LONG triggered: low={cur_low:.2f} <= stop={stop_price:.2f}")
                order = self.executor.place_market_order(
                    self.symbol, 'sell', self.pos_long.size,
                    'LONG', reduce_only=True)
                if order is None:
                    logger.warning(f"Stop loss market order failed for LONG "
                                   f"(size={self.pos_long.size}), skipping internal close")
                else:
                    actual_price = float(order.get('average', stop_price) or stop_price)
                    fee = self.pos_long.size * actual_price * fee_rate
                    pnl = self.pos_long.close_all(actual_price, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_long'] += 1
                    self._partial_stop_fired_long = False
                    self.grid_needs_regen = True
                    # ─── Stop-cooldown counter (#2) ───────────────────────
                    self._consecutive_stops += 1
                    cooldown_thresh = self.config.get('stop_cooldown_thresh', 2)
                    cooldown_bars = self.config.get('stop_cooldown_bars', 48)
                    if self._consecutive_stops >= cooldown_thresh:
                        self._descale_until_bar = self.bar_count + cooldown_bars
                        logger.info(f"  [SL]  Consecutive stops={self._consecutive_stops} ≥ "
                                    f"{cooldown_thresh}: de-scaling for {cooldown_bars} bars")
                    self._log_trade_event(
                        cur_time, actual_price, 0, LABEL_STOP_LONG, prev_regime, pnl)

        if not self.pos_long.is_open:
            self._partial_stop_fired_long = False

        if self.pos_short.is_open:
            effective_mult = atr_sl_mult * 1.5 if self._partial_stop_fired_short else atr_sl_mult
            stop_price = self.pos_short.avg_entry + effective_mult * prev_atr
            distance_pct = (stop_price - cur_high) / stop_price * 100
            
            stage1_trigger = cur_close >= stop_price
            stage2_trigger = cur_high >= stop_price

            logger.info(f"  [SL] SHORT stop={stop_price:.2f}  high={cur_high:.2f} close={cur_close:.2f} "
                        f"distance={distance_pct:+.2f}%{'  ← STAGE2' if stage2_trigger else ''}{' ← STAGE1' if stage1_trigger else ''}")

            if stage1_trigger and self.pos_short.num_fills > 1 and not self._partial_stop_fired_short:
                # Stage 1: partial stop — close 50% on candle CLOSE breach (anti-wick)
                close_qty = self.pos_short.size * 0.5
                logger.info(f"PARTIAL STOP SHORT (50%): close={cur_close:.2f} >= stop={stop_price:.2f}, "
                            f"closing {close_qty:.6f} of {self.pos_short.size:.6f}")
                order = self.executor.place_market_order(
                    self.symbol, 'buy', close_qty,
                    'SHORT', reduce_only=True)
                if order is None:
                    logger.warning(f"Partial stop loss market order failed for SHORT")
                else:
                    actual_price = float(order.get('average', stop_price) or stop_price)
                    fee = close_qty * actual_price * fee_rate
                    pnl = self.pos_short.close_fill(actual_price, close_qty, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_short'] += 1
                    self._partial_stop_fired_short = True
                    self.grid_needs_regen = True
                    # ─── Stop-cooldown counter (matches backtest) ─────────
                    self._consecutive_stops += 1
                    cooldown_thresh = self.config.get('stop_cooldown_thresh', 2)
                    cooldown_bars = self.config.get('stop_cooldown_bars', 48)
                    if self._consecutive_stops >= cooldown_thresh:
                        self._descale_until_bar = self.bar_count + cooldown_bars
                        logger.info(f"  [SL]  Consecutive stops={self._consecutive_stops} ≥ "
                                    f"{cooldown_thresh}: de-scaling for {cooldown_bars} bars")
                    self._log_trade_event(
                        cur_time, actual_price, close_qty, LABEL_STOP_SHORT, prev_regime, pnl)
            elif stage2_trigger:
                # Stage 2 (or single fill): close everything
                logger.info(f"STOP LOSS SHORT triggered: high={cur_high:.2f} >= stop={stop_price:.2f}")
                order = self.executor.place_market_order(
                    self.symbol, 'buy', self.pos_short.size,
                    'SHORT', reduce_only=True)
                if order is None:
                    logger.warning(f"Stop loss market order failed for SHORT "
                                   f"(size={self.pos_short.size}), skipping internal close")
                else:
                    actual_price = float(order.get('average', stop_price) or stop_price)
                    fee = self.pos_short.size * actual_price * fee_rate
                    pnl = self.pos_short.close_all(actual_price, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_short'] += 1
                    self._partial_stop_fired_short = False
                    self.grid_needs_regen = True
                    # ─── Stop-cooldown counter (#2) ───────────────────────
                    self._consecutive_stops += 1
                    cooldown_thresh = self.config.get('stop_cooldown_thresh', 2)
                    cooldown_bars = self.config.get('stop_cooldown_bars', 48)
                    if self._consecutive_stops >= cooldown_thresh:
                        self._descale_until_bar = self.bar_count + cooldown_bars
                        logger.info(f"  [SL]  Consecutive stops={self._consecutive_stops} ≥ "
                                    f"{cooldown_thresh}: de-scaling for {cooldown_bars} bars")
                    self._log_trade_event(
                        cur_time, actual_price, 0, LABEL_STOP_SHORT, prev_regime, pnl)

        if not self.pos_short.is_open:
            self._partial_stop_fired_short = False

        # Reset consecutive stop counter when no stop fired this bar
        if (not self.pos_long.is_open or cur_low > (self.pos_long.avg_entry - atr_sl_mult * prev_atr)) and \
           (not self.pos_short.is_open or cur_high < (self.pos_short.avg_entry + atr_sl_mult * prev_atr)):
            self._consecutive_stops = 0

        # ─── 4. LIQUIDATION CHECK ────────────────────────────
        if self.pos_long.is_open and leverage > 1.0:
            liq_price = calculate_liquidation_price(
                self.pos_long.avg_entry, self.pos_long.size,
                self.wallet_balance, 1, leverage)
            if cur_close <= liq_price and liq_price > 0:
                logger.critical(f"LIQUIDATION LONG: price={cur_close:.2f} <= liq={liq_price:.2f}")
                order = self.executor.place_market_order(
                    self.symbol, 'sell', self.pos_long.size,
                    'LONG', reduce_only=True)
                if order is None:
                    logger.critical("Liquidation market order failed for LONG!")
                else:
                    pnl = self.pos_long.close_all(cur_close, 0)
                    self.wallet_balance += pnl
                    self.metrics['liquidations'] += 1
                    self.grid_needs_regen = True
                    self._log_trade_event(
                        cur_time, cur_close, 0, LABEL_LIQUIDATION, prev_regime, pnl)

        if self.pos_short.is_open and leverage > 1.0:
            liq_price = calculate_liquidation_price(
                self.pos_short.avg_entry, self.pos_short.size,
                self.wallet_balance, -1, leverage)
            if cur_close >= liq_price and liq_price > 0:
                logger.critical(f"LIQUIDATION SHORT: price={cur_close:.2f} >= liq={liq_price:.2f}")
                order = self.executor.place_market_order(
                    self.symbol, 'buy', self.pos_short.size,
                    'SHORT', reduce_only=True)
                if order is None:
                    logger.critical("Liquidation market order failed for SHORT!")
                else:
                    pnl = self.pos_short.close_all(cur_close, 0)
                    self.wallet_balance += pnl
                    self.metrics['liquidations'] += 1
                    self.grid_needs_regen = True
                    self._log_trade_event(
                        cur_time, cur_close, 0, LABEL_LIQUIDATION, prev_regime, pnl)

        # ─── 5. PRUNING (5 methods, both sides) ─────────────────
        grid_spacing = calculate_dynamic_spacing(
            prev_atr, spacing_k, spacing_floor, cur_close)
        logger.info(f"  [GRID] spacing={grid_spacing:.2f}  "
                    f"levels={grid_levels}  regen_needed={self.grid_needs_regen}")
        grid_profit_potential = grid_spacing * order_pct * self.wallet_balance / max(cur_close, 1)

        # Use candle timestamp for pruning age checks (matches backtest behavior).
        # Wall-clock time would cause fills to age slightly faster than in backtest.
        prune_time = cur_time

        for pos, acc_profit, side_key in [
            (self.pos_long, self.accumulated_profit_long, 'long'),
            (self.pos_short, self.accumulated_profit_short, 'short'),
        ]:
            if not pos.is_open:
                continue
            anchor = self.grid_anchor_long if pos.side == 1 else self.grid_anchor_short
            prune_idx, prune_label = run_pruning_cycle(
                pos, cur_close, prune_time, prev_kama, prev_atr,
                anchor, grid_spacing, grid_profit_potential,
                acc_profit, self.config)
            if prune_idx >= 0 and prune_idx < len(pos.fills):
                fill = pos.fills[prune_idx]
                logger.info(f"  [PRUNE] {side_key.upper()} {prune_label}: "
                            f"fill@{fill['price']:.2f}  qty={fill['qty']:.4f}  "
                            f"age={(prune_time - fill['timestamp'])/3.6e6:.1f}h")
                fill_qty = fill['qty']

                # Market close the specific fill
                close_side = 'sell' if pos.side == 1 else 'buy'
                pos_side = 'LONG' if pos.side == 1 else 'SHORT'
                order = self.executor.place_market_order(
                    self.symbol, close_side, fill_qty,
                    pos_side, reduce_only=True)

                if order is None:
                    logger.warning(
                        f"Prune market order failed for {fill_qty} {pos_side}, "
                        f"skipping internal close to avoid desync")
                    continue

                fee = fill_qty * cur_close * fee_rate
                pnl = pos.close_specific_fill(prune_idx, cur_close, fee)
                self.wallet_balance += pnl
                self.metrics['prune_count'] += 1
                self.metrics['prune_types'][prune_label] = \
                    self.metrics['prune_types'].get(prune_label, 0) + 1

                # Update profit buffer: consume on offset prune, decrement on any loss
                acc_attr = f'accumulated_profit_{side_key}'
                if prune_label == 'PRUNE_OFFSET':
                    setattr(self, acc_attr, max(0.0, acc_profit + pnl))
                elif pnl < 0:
                    setattr(self, acc_attr, max(0.0, acc_profit + pnl))

                self.grid_needs_regen = True
                self._log_trade_event(
                    cur_time, cur_close, fill_qty, prune_label, prev_regime, pnl)

        # ─── 6. TRAILING UP ───────────────────────────────────────
        if trailing_enabled and prev_regime in (REGIME_UPTREND, REGIME_BREAKOUT_UP):
            if prev_er > trailing_er:
                top_level = self.grid_anchor_long + grid_spacing * grid_levels
                logger.info(f"  [TRAIL] Uptrend ER={prev_er:.2f} — price={cur_close:.2f}  "
                            f"top_level={top_level:.2f}{'  → SHIFTING' if cur_close > top_level else ''}")
                if cur_close > top_level:
                    shift = cur_close - self.grid_anchor_long
                    self.grid_anchor_long += shift * 0.5
                    self.trailing_anchor = self.grid_anchor_long
                    self.grid_needs_regen = True
                    self.metrics['trailing_shifts'] += 1
                    self.trade_logger.log_event(
                        'TRAILING_UP',
                        f'Anchor shifted to {self.grid_anchor_long:.2f}')

        # ─── 7. VaR CONSTRAINT ────────────────────────────────────
        total_exposure = (self.pos_long.size + self.pos_short.size) * cur_close
        var_blocked = check_var_constraint(
            total_exposure, prev_vol, max_dd_pct,
            self._total_equity(cur_close))
        if var_blocked:
            logger.info(f"  [VaR]  BLOCKED — exposure=${total_exposure:.2f}  vol={prev_vol:.4f}  "
                        f"max_dd={max_dd_pct:.0%}")
            self.metrics['var_blocks'] += 1

        # ─── VaR Pre-emptive De-leverage (#6) ─────────────────────
        # When unrealized drawdown hits 75% of hard cap, force-prune
        # the biggest loser on each side before we breach the limit.
        current_equity   = self._total_equity(cur_close)
        initial_capital  = self.config.get('initial_capital', self.initial_capital)
        drawdown_pct     = (initial_capital - current_equity) / max(initial_capital, 1e-9)
        if drawdown_pct > max_dd_pct * 0.75:
            logger.info(f"  [VaR]  Pre-emptive de-leverage: DD={drawdown_pct:.1%} > "
                        f"{max_dd_pct*0.75:.1%} threshold")
            for pos, pos_side_str in [
                (self.pos_long,  'LONG'),
                (self.pos_short, 'SHORT'),
            ]:
                if not pos.fills:
                    continue
                # Find fill with worst unrealized loss
                worst_idx, worst_loss = -1, 0.0
                for fi, fill in enumerate(pos.fills):
                    loss = ((fill['price'] - cur_close) * fill['qty'] if pos.side == 1
                            else (cur_close - fill['price']) * fill['qty'])
                    if loss > worst_loss:
                        worst_loss, worst_idx = loss, fi
                if worst_idx < 0:
                    continue
                fill_qty = pos.fills[worst_idx]['qty']
                close_side = 'sell' if pos.side == 1 else 'buy'
                order = self.executor.place_market_order(
                    self.symbol, close_side, fill_qty, pos_side_str, reduce_only=True)
                if order:
                    actual_p = float(order.get('average', cur_close) or cur_close)
                    fee = fill_qty * actual_p * fee_rate
                    pnl = pos.close_specific_fill(worst_idx, actual_p, fee)
                    self.wallet_balance += pnl
                    self.metrics['prune_count'] += 1
                    self.metrics['prune_types'][LABEL_PRUNE_VAR_WARNING] = \
                        self.metrics['prune_types'].get(LABEL_PRUNE_VAR_WARNING, 0) + 1
                    self.grid_needs_regen = True
                    logger.info(f"  [VaR]  Pruned worst {pos_side_str} fill "
                                f"@{fill_qty} pnl={pnl:.2f}")
                    self._log_trade_event(
                        cur_time, actual_p, fill_qty, LABEL_PRUNE_VAR_WARNING, prev_regime, pnl)

        # ─── 8. GENERATE GRID + PLACE ORDERS ─────────────────────
        if self.grid_needs_regen and not var_blocked:
            logger.info(f"  [GRID] Regenerating — reason: regen_flag set, VaR clear")
            # Cancel existing orders and re-fetch to catch race condition fills
            cancel_result = self.executor.cancel_all_orders(self.symbol)
            if cancel_result < 0:
                logger.error("Failed to cancel orders, skipping grid regeneration")
                return
            self.active_orders.clear()
            self.conditional_orders.clear()
            time.sleep(0.5)  # Brief pause for cancel to propagate

            # Check for fills during cancel
            self._sync_positions_from_exchange()

            # Fetch real funding rate
            funding_rate = self.executor.get_funding_rate(self.symbol)
            long_bias, short_bias = should_bias_for_funding(prev_regime, funding_rate)
            logger.info(f"  [GRID] funding_rate={funding_rate*100:.4f}%  "
                        f"long_bias={long_bias}  short_bias={short_bias}")

            # Read A-S time horizon from config (default 96 = 1 day of 15m bars)
            as_time_horizon = self.config.get('as_time_horizon', 96.0)

            # ─── Quarter-Kelly position sizing (#5) ────────────────────────
            kelly_mult = compute_kelly_fraction(self._kelly_trades, prev_regime)
            effective_order_pct = order_pct * kelly_mult

            # ─── Stop-loss de-scaling (#2) ──────────────────────────────
            if self.bar_count < self._descale_until_bar:
                effective_order_pct *= 0.5
                logger.info(f"  [SL]  De-scaling active: order_pct reduced by 50% "
                            f"(until bar {self._descale_until_bar}, now bar {self.bar_count})")

            # ─── Weekend/low-volume de-scaling (#7) ───────────────────────
            low_vol_threshold = self.config.get('low_volume_threshold', 0.5)
            cur_vol = indicators.get('cur_volume', 0.0)
            if self._vol_sma7 > 1e-9 and cur_vol < low_vol_threshold * self._vol_sma7:
                effective_max_pos_pct = max_position_pct * 0.7
                logger.info(f"  [VOL]  Low-volume bar (vol={cur_vol:.0f} < "
                            f"{low_vol_threshold:.0%} × SMA={self._vol_sma7:.0f}): "
                            f"max_pos reduced to {effective_max_pos_pct:.0%}")
            else:
                effective_max_pos_pct = max_position_pct

            self._place_grid_on_exchange(
                cur_close, prev_atr, prev_vol, prev_regime, funding_rate,
                cur_time, grid_levels, spacing_k, spacing_floor,
                gamma, kappa, effective_order_pct, leverage, max_inv_per_side,
                effective_max_pos_pct, allow_short, long_bias, short_bias,
                as_time_horizon, prev_er=prev_er)

            # ─── 8b. EXCHANGE-SIDE STOP LOSS (STOP_MARKET) ─────
            # Places STOP_MARKET orders on Binance so stops execute
            # in real-time between candles, not just at 15m checks.
            # The candle-based stop (step 3) remains as a backup.
            # If position is below exchange minimum order qty, skip SL
            # (executor will reject it anyway, but log here for clarity).
            min_qty = self.executor._min_amount(self.symbol) or 0.01

            if self.pos_long.is_open:
                sl_price = self.pos_long.avg_entry - atr_sl_mult * prev_atr
                if sl_price > 0 and self.pos_long.size >= min_qty:
                    sl_order = self.executor.place_stop_loss(
                        self.symbol, 'sell', self.pos_long.size,
                        sl_price, 'LONG',
                        client_order_id=f'sl_L_{int(cur_time)}')
                    if sl_order:
                        self.conditional_orders[str(sl_order['id'])] = {
                            'side': 'sell', 'price': sl_price,
                            'qty': float(sl_order.get('amount', self.pos_long.size)),
                            'direction': DIR_LONG, 'reduce_only': True,
                            'position_side': 'LONG',
                            'placed_at': int(cur_time),
                            'order_type': 'STOP_MARKET',
                        }
                elif sl_price > 0:
                    logger.info(f"Long position {self.pos_long.size} below min qty "
                                f"{min_qty}, skipping exchange SL (candle SL still active)")

            if self.pos_short.is_open and allow_short:
                sl_price = self.pos_short.avg_entry + atr_sl_mult * prev_atr
                if self.pos_short.size >= min_qty:
                    sl_order = self.executor.place_stop_loss(
                        self.symbol, 'buy', self.pos_short.size,
                        sl_price, 'SHORT',
                        client_order_id=f'sl_S_{int(cur_time)}')
                    if sl_order:
                        self.conditional_orders[str(sl_order['id'])] = {
                            'side': 'buy', 'price': sl_price,
                            'qty': float(sl_order.get('amount', self.pos_short.size)),
                            'direction': DIR_SHORT, 'reduce_only': True,
                            'position_side': 'SHORT',
                            'placed_at': int(cur_time),
                            'order_type': 'STOP_MARKET',
                        }
                else:
                    logger.info(f"Short position {self.pos_short.size} below min qty "
                                f"{min_qty}, skipping exchange SL (candle SL still active)")

            self.grid_needs_regen = False

        # ─── 10. FUNDING RATE ────────────────────────────────
        # In live trading, Binance applies funding to wallet automatically.
        # We only track it internally for metrics and pruning (funding_cost
        # per fill) but do NOT adjust wallet_balance — that would double-count.
        # Instead, refresh wallet_balance from exchange after funding.
        if is_funding_interval(self.bar_count):
            real_rate = self.executor.get_funding_rate(self.symbol)
            if self.pos_long.is_open:
                f_pnl = apply_funding(self.pos_long.size, cur_close, real_rate, 1)
                self.pos_long.add_funding(f_pnl)
                self.metrics['funding_pnl'] += f_pnl

            if self.pos_short.is_open:
                f_pnl = apply_funding(self.pos_short.size, cur_close, real_rate, -1)
                self.pos_short.add_funding(f_pnl)
                self.metrics['funding_pnl'] += f_pnl

            # Refresh wallet balance from exchange to pick up funding
            # and stay in sync with Binance's actual balance
            try:
                balance = self.executor.get_balance()
                ex_balance = balance.get('total', self.wallet_balance)
                if abs(ex_balance - self.wallet_balance) > 0.01:
                    logger.info(f"Wallet balance synced: "
                                f"${self.wallet_balance:.2f} → ${ex_balance:.2f} "
                                f"(diff: ${ex_balance - self.wallet_balance:+.2f})")
                    self.wallet_balance = ex_balance
            except Exception as e:
                logger.warning(f"Failed to refresh balance from exchange: {e}")

        # ─── 11. LOG EQUITY ──────────────────────────────────
        self._update_equity(cur_close)
        self._log_equity_bar(cur_time, cur_close, prev_regime)

    # ─── Grid Placement ──────────────────────────────────────────

    def _place_grid_on_exchange(self, price, atr, vol, regime, funding_rate,
                                 timestamp, grid_levels, spacing_k, spacing_floor,
                                 gamma, kappa, order_pct, leverage,
                                 max_inv_per_side, max_position_pct, allow_short,
                                 long_bias, short_bias,
                                 as_time_horizon=96.0,
                                 prev_er=0.5):
        """Generate and place grid orders on exchange."""
        # Adaptive floor: scales with coin vol ratio (matches backtest)
        vol_scale = self.config.get('adaptive_floor_scale', 1.5)
        adaptive_floor = calculate_adaptive_floor(atr, price, spacing_floor, vol_scale)
        adaptive_floor_pct = adaptive_floor / price if price > 1e-9 else spacing_floor
        base_spacing = calculate_dynamic_spacing(atr, spacing_k, adaptive_floor_pct, price)

        # ─── Asymmetric Grid Spacing based on Regime & ER (matches backtest) ──
        long_spacing_mult = 1.0
        short_spacing_mult = 1.0
        if regime in (REGIME_UPTREND, REGIME_BREAKOUT_UP):
            long_spacing_mult = max(0.5, 1.0 - (prev_er / 2.0))
            short_spacing_mult = 1.0 + prev_er
        elif regime in (REGIME_DOWNTREND, REGIME_BREAKOUT_DOWN):
            long_spacing_mult = 1.0 + prev_er
            short_spacing_mult = max(0.5, 1.0 - (prev_er / 2.0))

        base_long_spacing = base_spacing * long_spacing_mult
        base_short_spacing = base_spacing * short_spacing_mult
        self._last_grid_spacing = base_spacing

        # ─── LONG GRID ──────────────────────────────────────
        can_open_long_entry = regime not in (REGIME_DOWNTREND, REGIME_BREAKOUT_DOWN)

        inv_q_long = normalize_inventory(
            self.pos_long.size,
            max_inv_per_side * order_pct * self.wallet_balance / max(price, 1))
        buy_sp, sell_sp, anchor_long = get_skewed_grid_params(
            price, inv_q_long, gamma, vol, kappa,
            base_long_spacing, base_long_spacing,
            time_horizon=as_time_horizon)

        self.grid_anchor_long = anchor_long

        # ─── Geometric vs arithmetic grid levels (#4) ────────────
        grid_mode = self.config.get('grid_mode', 'geometric')
        if grid_mode == 'geometric':
            buy_pct  = (buy_sp * long_bias) / max(price, 1e-9)
            sell_pct = sell_sp / max(price, 1e-9)
            buy_levels, sell_levels = generate_geometric_grid_levels(
                anchor_long, buy_pct, sell_pct, grid_levels)
        else:
            buy_levels, sell_levels = generate_grid_levels(
                anchor_long, buy_sp * long_bias, sell_sp, grid_levels)

        # ─── Funding harvest sizing tilt (#8) ─────────────────────
        # In NOISE with strong funding: tilt order size toward receiving side
        fund_order_pct_long  = order_pct
        fund_order_pct_short = order_pct
        funding_harvest_threshold = self.config.get('funding_harvest_threshold', 0.0002)
        if regime == REGIME_NOISE and abs(funding_rate) > funding_harvest_threshold:
            tilt = 0.25 * order_pct
            if funding_rate > 0:            # Positive: shorts receive funding
                fund_order_pct_short = min(order_pct + tilt, order_pct * 1.5)
                fund_order_pct_long  = max(order_pct - tilt, order_pct * 0.5)
                logger.info(f"  [FUND] Positive funding {funding_rate*100:.4f}%: "
                            f"short tilt +{tilt:.4%}, long tilt -{tilt:.4%}")
            else:                           # Negative: longs receive funding
                fund_order_pct_long  = min(order_pct + tilt, order_pct * 1.5)
                fund_order_pct_short = max(order_pct - tilt, order_pct * 0.5)
                logger.info(f"  [FUND] Negative funding {funding_rate*100:.4f}%: "
                            f"long tilt +{tilt:.4%}, short tilt -{tilt:.4%}")

        long_orders = 0

        # Place long buy entries
        if can_open_long_entry and self.pos_long.num_fills < max_inv_per_side:
            max_notional = self.wallet_balance * max_position_pct
            current_notional = self.pos_long.size * price
            for lvl_i, lvl_price in enumerate(buy_levels):
                if lvl_price <= 0 or current_notional >= max_notional:
                    break
                qty = calculate_order_qty(self.wallet_balance, lvl_price, fund_order_pct_long, leverage)
                if qty < 1e-12:
                    continue
                order = self.executor.place_limit_order(
                    self.symbol, 'buy', qty, lvl_price, 'LONG',
                    reduce_only=False,
                    client_order_id=f'grid_L_BUY_{lvl_i}_{int(timestamp)}')
                if order:
                    actual_qty = float(order.get('amount', qty))
                    self.active_orders[str(order['id'])] = {
                        'side': 'buy', 'price': lvl_price, 'qty': actual_qty,
                        'direction': DIR_LONG, 'reduce_only': False,
                        'grid_level': lvl_i, 'position_side': 'LONG',
                        'placed_at': int(timestamp),
                    }
                    current_notional += actual_qty * lvl_price
                    long_orders += 1

        # Place long TPs as TAKE_PROFIT_MARKET conditional orders
        # Concentrated: tp_concentration % at nearest level, rest spread across remaining
        # If per-level qty is below exchange minimum, aggregate into a single TP.
        if self.pos_long.is_open:
            min_qty = self.executor._min_amount(self.symbol) or 0.01
            tp_concentration = self.config.get('tp_concentration', 0.6)
            n_levels = max(grid_levels, 1)

            # Check if the smallest TP slice would be below exchange minimum
            smallest_tp = self.pos_long.size * (1.0 - tp_concentration) / max(n_levels - 1, 1)
            if smallest_tp < min_qty:
                # Position too small to split — place single TP at first sell level
                if self.pos_long.size >= min_qty and len(sell_levels) > 0:
                    order = self.executor.place_take_profit(
                        self.symbol, 'sell', self.pos_long.size, sell_levels[0], 'LONG',
                        client_order_id=f'grid_L_TP_agg_{int(timestamp)}')
                    if order:
                        actual_qty = float(order.get('amount', self.pos_long.size))
                        self.conditional_orders[str(order['id'])] = {
                            'side': 'sell', 'price': sell_levels[0], 'qty': actual_qty,
                            'direction': DIR_LONG, 'reduce_only': True,
                            'grid_level': 0, 'position_side': 'LONG',
                            'placed_at': int(timestamp),
                            'order_type': 'TAKE_PROFIT_MARKET',
                        }
                        long_orders += 1
            else:
                for lvl_i, lvl_price in enumerate(sell_levels):
                    if lvl_i == 0:
                        tp_qty = self.pos_long.size * tp_concentration
                    else:
                        remaining_levels = max(n_levels - 1, 1)
                        tp_qty = self.pos_long.size * (1.0 - tp_concentration) / remaining_levels
                    if tp_qty < 1e-12:
                        break
                    order = self.executor.place_take_profit(
                        self.symbol, 'sell', tp_qty, lvl_price, 'LONG',
                        client_order_id=f'grid_L_TP_{lvl_i}_{int(timestamp)}')
                    if order:
                        actual_qty = float(order.get('amount', tp_qty))
                        self.conditional_orders[str(order['id'])] = {
                            'side': 'sell', 'price': lvl_price, 'qty': actual_qty,
                            'direction': DIR_LONG, 'reduce_only': True,
                            'grid_level': lvl_i, 'position_side': 'LONG',
                            'placed_at': int(timestamp),
                            'order_type': 'TAKE_PROFIT_MARKET',
                        }
                        long_orders += 1

        if long_orders > 0:
            self.trade_logger.log_grid_placement(
                'LONG', long_orders,
                (float(buy_levels[-1]) if len(buy_levels) > 0 else price,
                 float(sell_levels[-1]) if len(sell_levels) > 0 else price))
            if self.tg:
                buy_range = (f"${float(buy_levels[-1]):,.2f} – ${float(buy_levels[0]):,.2f}"
                             if len(buy_levels) > 0 else 'none')
                sell_range = (f"${float(sell_levels[0]):,.2f} – ${float(sell_levels[-1]):,.2f}"
                              if len(sell_levels) > 0 else 'none')
                self.tg.send(
                    f'📋 <b>LONG GRID PLACED</b> — <code>{self.symbol}</code>\n'
                    f'Price:    <code>${price:,.2f}</code>  ATR=<code>{atr:.2f}</code>\n'
                    f'Buys:     <code>{buy_range}</code>\n'
                    f'TPs:      <code>{sell_range}</code>\n'
                    f'Orders:   <code>{long_orders}</code>  spacing=<code>{base_spacing:.2f}</code>',
                    silent=True)

        # ─── SHORT GRID ─────────────────────────────────────
        if not allow_short:
            return

        can_open_short_entry = regime not in (REGIME_UPTREND, REGIME_BREAKOUT_UP)

        inv_q_short = normalize_inventory(
            -self.pos_short.size,
            max_inv_per_side * order_pct * self.wallet_balance / max(price, 1))
        buy_sp_s, sell_sp_s, anchor_short = get_skewed_grid_params(
            price, inv_q_short, gamma, vol, kappa,
            base_short_spacing, base_short_spacing,
            time_horizon=as_time_horizon)

        self.grid_anchor_short = anchor_short
        
        if grid_mode == 'geometric':
            buy_pct_s  = buy_sp_s / max(price, 1e-9)
            sell_pct_s = (sell_sp_s * short_bias) / max(price, 1e-9)
            buy_levels_s, sell_levels_s = generate_geometric_grid_levels(
                anchor_short, buy_pct_s, sell_pct_s, grid_levels)
        else:
            buy_levels_s, sell_levels_s = generate_grid_levels(
                anchor_short, buy_sp_s, sell_sp_s * short_bias, grid_levels)


        short_orders = 0

        # Short sell entries
        if can_open_short_entry and self.pos_short.num_fills < max_inv_per_side:
            max_notional = self.wallet_balance * max_position_pct
            current_notional = self.pos_short.size * price
            for lvl_i, lvl_price in enumerate(sell_levels_s):
                if current_notional >= max_notional:
                    break
                qty = calculate_order_qty(self.wallet_balance, lvl_price, fund_order_pct_short, leverage)
                if qty < 1e-12:
                    continue
                order = self.executor.place_limit_order(
                    self.symbol, 'sell', qty, lvl_price, 'SHORT',
                    reduce_only=False,
                    client_order_id=f'grid_S_SELL_{lvl_i}_{int(timestamp)}')
                if order:
                    actual_qty = float(order.get('amount', qty))
                    self.active_orders[str(order['id'])] = {
                        'side': 'sell', 'price': lvl_price, 'qty': actual_qty,
                        'direction': DIR_SHORT, 'reduce_only': False,
                        'grid_level': lvl_i, 'position_side': 'SHORT',
                        'placed_at': int(timestamp),
                    }
                    current_notional += actual_qty * lvl_price
                    short_orders += 1

        # Short TPs as TAKE_PROFIT_MARKET conditional orders
        # Concentrated: tp_concentration % at nearest level, rest spread across remaining
        # If per-level qty is below exchange minimum, aggregate into a single TP.
        if self.pos_short.is_open:
            min_qty = self.executor._min_amount(self.symbol) or 0.01
            tp_concentration = self.config.get('tp_concentration', 0.6)
            n_levels = max(grid_levels, 1)

            smallest_tp = self.pos_short.size * (1.0 - tp_concentration) / max(n_levels - 1, 1)
            if smallest_tp < min_qty:
                # Position too small to split — place single TP at first buy level
                if self.pos_short.size >= min_qty and len(buy_levels_s) > 0:
                    order = self.executor.place_take_profit(
                        self.symbol, 'buy', self.pos_short.size, buy_levels_s[0], 'SHORT',
                        client_order_id=f'grid_S_TP_agg_{int(timestamp)}')
                    if order:
                        actual_qty = float(order.get('amount', self.pos_short.size))
                        self.conditional_orders[str(order['id'])] = {
                            'side': 'buy', 'price': buy_levels_s[0], 'qty': actual_qty,
                            'direction': DIR_SHORT, 'reduce_only': True,
                            'grid_level': 0, 'position_side': 'SHORT',
                            'placed_at': int(timestamp),
                            'order_type': 'TAKE_PROFIT_MARKET',
                        }
                        short_orders += 1
            else:
                for lvl_i, lvl_price in enumerate(buy_levels_s):
                    if lvl_i == 0:
                        tp_qty = self.pos_short.size * tp_concentration
                    else:
                        remaining_levels = max(n_levels - 1, 1)
                        tp_qty = self.pos_short.size * (1.0 - tp_concentration) / remaining_levels
                    if tp_qty < 1e-12 or lvl_price <= 0:
                        break
                    order = self.executor.place_take_profit(
                        self.symbol, 'buy', tp_qty, lvl_price, 'SHORT',
                        client_order_id=f'grid_S_TP_{lvl_i}_{int(timestamp)}')
                    if order:
                        actual_qty = float(order.get('amount', tp_qty))
                        self.conditional_orders[str(order['id'])] = {
                            'side': 'buy', 'price': lvl_price, 'qty': actual_qty,
                            'direction': DIR_SHORT, 'reduce_only': True,
                            'grid_level': lvl_i, 'position_side': 'SHORT',
                            'placed_at': int(timestamp),
                            'order_type': 'TAKE_PROFIT_MARKET',
                        }
                        short_orders += 1

        if short_orders > 0:
            self.trade_logger.log_grid_placement(
                'SHORT', short_orders,
                (float(sell_levels_s[-1]) if len(sell_levels_s) > 0 else price,
                 float(buy_levels_s[-1]) if len(buy_levels_s) > 0 else price))
            if self.tg:
                sell_range = (f"${float(sell_levels_s[0]):,.2f} – ${float(sell_levels_s[-1]):,.2f}"
                              if len(sell_levels_s) > 0 else 'none')
                buy_range = (f"${float(buy_levels_s[-1]):,.2f} – ${float(buy_levels_s[0]):,.2f}"
                             if len(buy_levels_s) > 0 else 'none')
                self.tg.send(
                    f'📋 <b>SHORT GRID PLACED</b> — <code>{self.symbol}</code>\n'
                    f'Price:    <code>${price:,.2f}</code>  ATR=<code>{atr:.2f}</code>\n'
                    f'Sells:    <code>{sell_range}</code>\n'
                    f'TPs:      <code>{buy_range}</code>\n'
                    f'Orders:   <code>{short_orders}</code>  spacing=<code>{base_spacing:.2f}</code>',
                    silent=True)

    # ─── Equity Helpers ──────────────────────────────────────────

    def _update_equity(self, price: float):
        self.pos_long.update_unrealized(price)
        self.pos_short.update_unrealized(price)

    def _total_equity(self, price: float) -> float:
        self._update_equity(price)
        return (self.wallet_balance
                + self.pos_long.unrealized_pnl
                + self.pos_short.unrealized_pnl)

    # ─── Logging Helpers ─────────────────────────────────────────

    def _log_trade_event(self, timestamp, price, qty, label, regime, pnl):
        """Log and persist a trade event."""
        trade = {
            'timestamp': int(timestamp),
            'datetime': datetime.now(timezone.utc).isoformat(),
            'symbol': self.symbol,
            'price': price,
            'qty': qty,
            'label': label,
            'regime': regime,
            'pnl': pnl,
            'exchange_order_id': '',
            'fill_price': price,
            'fee': 0.0,
            'wallet_balance_after': self.wallet_balance,
            'equity_after': self._total_equity(price),
            'pos_long_size': self.pos_long.size,
            'pos_short_size': self.pos_short.size,
        }
        self.trade_logger.log_trade(trade)
        self.state.save_trade(trade)

        # Update Kelly trade memory for dynamic sizing (#5)
        if pnl != 0:
            self._kelly_trades.append({
                'pnl': pnl,
                'regime': regime,
            })
            if len(self._kelly_trades) > self.config.get('kelly_window', 50):
                self._kelly_trades = self._kelly_trades[-self.config.get('kelly_window', 50):]

        # Telegram: notify on every trade event
        if self.tg:
            regime_names = {0: 'NOISE', 1: 'UPTREND', -1: 'DOWNTREND',
                            2: 'BREAKOUT_UP', -2: 'BREAKOUT_DOWN'}
            regime_str = regime_names.get(regime, str(regime))
            equity = trade['equity_after']
            # Stops/CB/liquidations are loud (sound on); fills are silent
            is_loud = any(k in label for k in ('STOP', 'CIRCUIT', 'LIQUIDATION'))
            self.tg.send(
                fmt_trade(label, self.symbol, price, qty, pnl, regime_str, equity),
                silent=not is_loud)

    def _log_equity_bar(self, timestamp, price, regime):
        """Log equity snapshot for this bar."""
        self.trade_logger.log_equity(
            int(timestamp), self.wallet_balance,
            self.pos_long.unrealized_pnl,
            self.pos_short.unrealized_pnl,
            self.pos_long.size, self.pos_short.size,
            regime)

    # ─── State Serialization ─────────────────────────────────────

    def _serialize_state(self) -> dict:
        """Serialize current state for persistence."""
        def _serialize_pos(pos):
            return {
                'size': pos.size,
                'avg_entry': pos.avg_entry,
                'realized_pnl': pos.realized_pnl,
                'unrealized_pnl': pos.unrealized_pnl,
                'funding_pnl': pos.funding_pnl,
                'num_fills': pos.num_fills,
                'fills': [
                    {'price': f['price'], 'qty': f['qty'],
                     'timestamp': f['timestamp'], 'funding_cost': f['funding_cost']}
                    for f in pos.fills
                ],
            }

        return {
            'version': 'v4.0',
            'symbol': self.symbol,
            'timestamp': int(time.time() * 1000),
            'datetime': datetime.now(timezone.utc).isoformat(),
            'wallet_balance': self.wallet_balance,
            'initial_capital': self.initial_capital,
            'pos_long': _serialize_pos(self.pos_long),
            'pos_short': _serialize_pos(self.pos_short),
            'halted': self.halted,
            'grid_anchor_long': self.grid_anchor_long,
            'grid_anchor_short': self.grid_anchor_short,
            'trailing_anchor': self.trailing_anchor,
            'grid_needs_regen': self.grid_needs_regen,
            'accumulated_profit_long': self.accumulated_profit_long,
            'accumulated_profit_short': self.accumulated_profit_short,
            'bar_count': self.bar_count,
            'last_processed_bar_ts': self.last_processed_bar_ts,
            'active_orders': self.active_orders,
            'conditional_orders': self.conditional_orders,
            'metrics': self.metrics,
            '_kelly_trades': self._kelly_trades,
            '_consecutive_stops': self._consecutive_stops,
            '_descale_until_bar': self._descale_until_bar,
            '_vol_history': self._vol_history,
            '_partial_stop_fired_long': self._partial_stop_fired_long,
            '_partial_stop_fired_short': self._partial_stop_fired_short,
        }

    # ─── Shutdown ────────────────────────────────────────────────

    def shutdown(self):
        """Graceful shutdown: cancel orders, save state, keep positions."""
        if not self._running:
            return  # Already shutting down (idempotent guard)
        self._running = False
        logger.info("Initiating graceful shutdown...")

        # Cancel all open orders
        self.executor.cancel_all_orders(self.symbol)
        self.active_orders.clear()
        self.conditional_orders.clear()

        # Save final state
        self.state.save(self._serialize_state())
        self.state.save_snapshot(self._serialize_state(), label='shutdown')

        # Log session summary
        if self._session_start:
            duration = (datetime.now(timezone.utc) - self._session_start).total_seconds() / 3600
        else:
            duration = 0
        self.trade_logger.log_session_summary(self.metrics, duration)

        logger.info("Shutdown complete. Positions left open on exchange.")

    def emergency_shutdown(self):
        """Emergency shutdown: cancel orders AND close all positions."""
        self._running = False
        logger.critical("EMERGENCY SHUTDOWN — closing all positions")

        # Cancel orders
        self.executor.cancel_all_orders(self.symbol)
        self.active_orders.clear()
        self.conditional_orders.clear()

        # Get current price for PnL calculation
        ticker = self.executor.get_ticker(self.symbol)
        price = ticker.get('last', 0)
        if price <= 0:
            # Fallback to last known price from buffer
            if self.candle_buffer is not None and len(self.candle_buffer) > 0:
                price = float(self.candle_buffer['close'].iloc[-1])

        fee_rate = self.config.get('fee_taker', 0.0002)

        # Close long position
        if self.pos_long.is_open:
            self.executor.place_market_order(
                self.symbol, 'sell', self.pos_long.size,
                'LONG', reduce_only=True)
            fee = self.pos_long.size * price * fee_rate
            pnl = self.pos_long.close_all(price, fee)
            self.wallet_balance += pnl

        # Close short position
        if self.pos_short.is_open:
            self.executor.place_market_order(
                self.symbol, 'buy', self.pos_short.size,
                'SHORT', reduce_only=True)
            fee = self.pos_short.size * price * fee_rate
            pnl = self.pos_short.close_all(price, fee)
            self.wallet_balance += pnl

        # Save state
        self.state.save(self._serialize_state())
        self.state.save_snapshot(self._serialize_state(), label='emergency')

        if self._session_start:
            duration = (datetime.now(timezone.utc) - self._session_start).total_seconds() / 3600
        else:
            duration = 0
        self.trade_logger.log_session_summary(self.metrics, duration)

        logger.critical("Emergency shutdown complete. All positions closed.")
