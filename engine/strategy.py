"""
GridStrategyV4 Main Orchestrator.

Hedge Mode long/short grid strategy for perpetual futures.
Implements the full V4 main loop per CLAUDE.md spec.

Main Loop Order:
  1. Regime Detection (KAMA/ER) → NOISE, UPTREND, DOWNTREND, BREAKOUT
  2. Circuit Breaker → Halt if Z < -3
  3. Stop Loss → ATR-based per side
  4. Pruning → 5 methods (per side)
  5. Trailing Up → Shift grid if price exceeds top
  6. Generate Grid → Adaptive spacing + inventory skew
  7. Execute Fills → Direction-aware ordering (Hedge Mode)
  8. Funding Rate → Apply/track per position
  9. Log Equity → With regime label
"""
import numpy as np

from core.kama import (
    calculate_er, calculate_kama, detect_regime,
    apply_regime_hysteresis, resample_ohlcv,
    REGIME_NOISE, REGIME_UPTREND, REGIME_DOWNTREND,
    REGIME_BREAKOUT_UP, REGIME_BREAKOUT_DOWN, REGIME_NAMES,
)
from core.atr import calculate_atr, calculate_z_score, calculate_rolling_volatility
from core.adx import calculate_adx
from core.grid import (
    generate_grid_levels, calculate_order_qty,
    calculate_dynamic_spacing, calculate_adaptive_floor,
    generate_geometric_grid_levels,
)
from core.inventory import get_skewed_grid_params, normalize_inventory
from core.risk import (
    calculate_liquidation_price, calculate_var_95, check_var_constraint,
    calculate_unrealized_pnl, calculate_funding_pnl,
)
from core.funding import (
    generate_synthetic_funding_rates, is_funding_interval,
    apply_funding, should_bias_for_funding,
)
from core.kelly import compute_kelly_fraction
from engine.types import (
    PositionTracker, SIDE_BUY, SIDE_SELL, DIR_LONG, DIR_SHORT,
    LABEL_BUY_OPEN_LONG, LABEL_SELL_CLOSE_LONG,
    LABEL_SELL_OPEN_SHORT, LABEL_BUY_CLOSE_SHORT,
    LABEL_STOP_LONG, LABEL_STOP_SHORT,
    LABEL_CIRCUIT_BREAKER, LABEL_LIQUIDATION,
    LABEL_PRUNE_OLDEST, LABEL_PRUNE_DEVIANCE, LABEL_PRUNE_GAP,
    LABEL_PRUNE_FUNDING, LABEL_PRUNE_OFFSET, LABEL_PRUNE_VAR_WARNING,
)
from engine.matching import OrderBook
from engine.pruning import run_pruning_cycle
from core.ml_regime import classify_regimes_gmm
from config import BACKTEST_FILL_CONF


class GridStrategyV4:
    """
    Volatility-Adaptive Asymmetric Grid Trading Engine.

    Hedge Mode: simultaneous independent long + short grids.
    """

    def __init__(self, config: dict):
        self.config = config

        # Capital
        self.initial_capital = config['initial_capital']
        self.wallet_balance = self.initial_capital

        # Hedge Mode positions
        self.pos_long = PositionTracker(side=1)
        self.pos_short = PositionTracker(side=-1)

        # Order book
        self.order_book = OrderBook(max_orders=config.get('max_orders', 500))

        # Grid state
        self.grid_anchor_long = 0.0
        self.grid_anchor_short = 0.0
        self.grid_needs_regen = True
        self._last_grid_spacing = 0.0
        self._prev_regime = 0

        # Partial stop loss state: tracks if first-stage stop already fired
        self._partial_stop_fired_long = False
        self._partial_stop_fired_short = False

        # Prune cooldown: track last prune bar per side
        self._last_prune_bar_long = -999
        self._last_prune_bar_short = -999

        # Trailing state
        self.trailing_anchor = 0.0

        # Circuit breaker state
        self.halted = False

        # Accumulated grid profit for profit-offset pruning (per side)
        # Decremented on losses so buffer reflects net earned, not gross wins
        self.accumulated_profit_long = 0.0
        self.accumulated_profit_short = 0.0

        # Metrics counters
        self.metrics = {
            'longs_opened': 0,
            'longs_closed': 0,
            'shorts_opened': 0,
            'shorts_closed': 0,
            'stops_long': 0,
            'stops_short': 0,
            'prune_count': 0,
            'prune_types': {},
            'circuit_breaker_triggers': 0,
            'trailing_shifts': 0,
            'funding_pnl': 0.0,
            'var_blocks': 0,
            'liquidations': 0,
        }

        # ─── Stop-loss cooldown / de-scaling (#2) ────────────
        self._consecutive_stops = 0
        self._last_stop_bar = -999
        self._descale_until_bar = -1   # bar index until which order_pct is halved
        self._STOP_COOLDOWN_BARS = 48  # 12 hours on 15m
        self._STOP_COOLDOWN_THRESH = 2 # stops within window before de-scaling

        # ─── Kelly rolling trade history (#5) ─────────────────
        self._kelly_trades: list = []   # rolling window of {pnl, regime} dicts
        self._KELLY_WINDOW = 50

    def prepare_indicators(self, df) -> dict:
        """
        Pre-compute all indicators from OHLCV DataFrame.

        Uses prev-bar indicators for decisions (no lookahead).

        Multi-timeframe regime: regime is computed on a higher timeframe
        (regime_timeframe_mult × 15m, default 4 = 1H) to eliminate 15m
        noise-driven regime flips, then expanded back to 15m resolution.

        Regime hysteresis: after HTF regime detection, a debounce filter
        (regime_hysteresis_bars, default 3) prevents single-bar transitions.
        """
        closes = df['close'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        opens = df['open'].values.astype(np.float64)
        timestamps = df['timestamp'].values.astype(np.float64)

        kama_p = self.config['kama_period']
        atr_p = self.config['atr_period']
        htf_mult = self.config.get('regime_timeframe_mult', 4)   # 4 = 1H on 15m
        hysteresis_bars = self.config.get('regime_hysteresis_bars', 3)

        # ─── 15m indicators ───────────────────────────────────────────
        atr = calculate_atr(highs, lows, closes, atr_p)
        z_score = calculate_z_score(closes, atr, 20)
        volatility = calculate_rolling_volatility(closes, atr_p)
        
        # ─── Funding Rates (Real vs Synthetic) ────────────────────────
        if 'fundingRate' in df.columns:
            funding_rates = df['fundingRate'].values.astype(np.float64)
        else:
            funding_rates = generate_synthetic_funding_rates(closes)

        # ─── Multi-timeframe regime (HTF KAMA/ER on resampled data) ───
        htf_close, htf_high, htf_low = resample_ohlcv(closes, highs, lows, mult=htf_mult)
        htf_atr = calculate_atr(htf_high, htf_low, htf_close, atr_p)
        er = calculate_er(htf_close, kama_p)
        kama = calculate_kama(htf_close, er, self.config['kama_fast'],
                              self.config['kama_slow'])

        # ─── ADX veto filter (#1) — computed on 15m for sensitivity ───
        adx = calculate_adx(highs, lows, closes, self.config.get('adx_period', 14))

        # ─── Regime Detection (ML vs KAMA) ─────────────────────────
        if self.config.get('use_ml_regime', False):
            raw_regime = classify_regimes_gmm(closes, highs, lows, df['volume'].values)
        else:
            raw_regime = detect_regime(kama, er, htf_atr,
                                       self.config['regime_threshold'],
                                       self.config['er_trend_threshold'])

        # ─── ADX Veto: override TREND → NOISE when ADX < threshold ──
        adx_trend_threshold = self.config.get('adx_trend_threshold', 25.0)
        veto_regime = raw_regime.copy()
        for idx in range(len(veto_regime)):
            if abs(veto_regime[idx]) in (1, 2) and adx[idx] < adx_trend_threshold:
                veto_regime[idx] = REGIME_NOISE

        # ─── Hysteresis debounce ──────────────────────────────────────
        regime = apply_regime_hysteresis(veto_regime, min_bars=hysteresis_bars)

        return {
            'open': opens, 'high': highs, 'low': lows, 'close': closes,
            'timestamp': timestamps,
            'kama': kama, 'er': er, 'atr': atr,
            'regime': regime, 'z_score': z_score,
            'volatility': volatility, 'funding_rates': funding_rates,
            'volume': df['volume'].values.astype(np.float64),
        }

    def run(self, df) -> dict:
        """
        Run the full backtest.

        Lookahead-free: bar i uses indicators from bar i-1,
        price action (OHLC) from bar i for fill detection.

        Returns dict with equity_curve, trades, metrics.
        """
        data = self.prepare_indicators(df)
        n = len(data['close'])

        equity_curve = np.zeros(n, dtype=np.float64)
        equity_curve[0] = self.initial_capital
        regime_log = np.zeros(n, dtype=np.int8)
        trades = []

        self.grid_anchor_long = data['close'][0]
        self.grid_anchor_short = data['close'][0]
        self.trailing_anchor = data['close'][0]

        leverage = self.config.get('leverage', 1.0)
        allow_short = self.config.get('allow_short', True)
        fee_maker = self.config.get('fee_maker', 0.0002)   # limit entries
        fee_taker = self.config.get('fee_taker', 0.0005)   # market exits
        fee_rate = fee_taker  # default for stops/prunes/liquidation
        order_pct = self.config['order_pct']
        grid_levels = self.config['grid_levels']
        gamma = self.config['gamma']
        kappa = self.config.get('kappa', 1.5)
        spacing_k = self.config['grid_spacing_k']
        spacing_floor = self.config.get('spacing_floor', 0.005)
        max_inv_per_side = self.config.get('max_inventory_per_side', 10)
        atr_sl_mult = self.config.get('atr_sl_mult', 3.5)
        max_position_pct = self.config.get('max_position_pct', 0.7)
        as_time_horizon = self.config.get('as_time_horizon', 96.0)

        trailing_enabled = self.config.get('trailing_enabled', True)
        trailing_er = self.config.get('trailing_activation_er', 0.65)
        cb_enabled = self.config.get('circuit_breaker_enabled', True)
        halt_z = self.config.get('halt_z_threshold', -3.0)
        resume_z = self.config.get('resume_z_threshold', -1.0)

        max_dd_pct = self.config.get('max_drawdown_pct', 0.15)

        # Simulation params
        slippage = BACKTEST_FILL_CONF.get('slippage_pct', 0.0005)
        fill_prob = BACKTEST_FILL_CONF.get('fill_probability', 1.0)

        # Reset accumulated profit accumulators for this backtest run
        self.accumulated_profit_long = 0.0
        self.accumulated_profit_short = 0.0

        # ─── Rolling volume for weekend/low-liquidity de-scaling (#7) ─
        vol_arr  = data.get('volume', np.ones(n))
        vol_sma7 = np.zeros(n, dtype=np.float64)   # 7-day SMA of volume
        _vol_window = 96 * 7  # 7 days of 15m bars
        for _vi in range(1, n):
            _start = max(0, _vi - _vol_window)
            vol_sma7[_vi] = vol_arr[_start:_vi].mean()

        # ─── Rolling 30-bar log-returns for Kelly (#5) ────────────────
        log_ret = np.zeros(n, dtype=np.float64)
        for _ri in range(1, n):
            if data['close'][_ri - 1] > 1e-9:
                log_ret[_ri] = np.log(data['close'][_ri] / data['close'][_ri - 1])

        for i in range(1, n):
            # ─── PREV BAR indicators (no lookahead) ──────────
            prev_kama = data['kama'][i - 1]
            prev_atr = data['atr'][i - 1]
            prev_regime = int(data['regime'][i - 1])
            prev_z = data['z_score'][i - 1]
            prev_er = data['er'][i - 1]
            prev_vol = data['volatility'][i - 1]
            prev_funding = data['funding_rates'][i - 1]

            # ─── CURRENT BAR price action ─────────────────────
            cur_open = data['open'][i]
            cur_high = data['high'][i]
            cur_low = data['low'][i]
            cur_close = data['close'][i]
            cur_time = data['timestamp'][i]

            regime_log[i] = prev_regime

            # ─── 1. REGIME (already computed) ─────────────────

            # ─── 2. CIRCUIT BREAKER ───────────────────────────
            if cb_enabled:
                if not self.halted and prev_z < halt_z:
                    self.halted = True
                    self.metrics['circuit_breaker_triggers'] += 1
                    # Cancel all opening orders, keep positions
                    self.order_book.cancel_all()
                    trades.append(self._trade_record(
                        i, cur_time, cur_close, 0.0, LABEL_CIRCUIT_BREAKER, prev_regime))

                if self.halted and prev_z > resume_z:
                    self.halted = False
                    self.grid_needs_regen = True

            if self.halted:
                # Still update PnL / equity while halted
                self._update_equity(cur_close)
                equity_curve[i] = self._total_equity(cur_close)
                continue

            # ─── 3. STOP LOSS (ATR-based) ─────────────────────
            stop_trades = self._check_stop_loss(
                cur_close, cur_low, cur_high, cur_time, i,
                prev_kama, prev_atr, atr_sl_mult, fee_rate, prev_regime, slippage)
            trades.extend(stop_trades)

            # ─── 4. LIQUIDATION CHECK ─────────────────────────
            liq_trades = self._check_liquidation(cur_close, cur_time, i, fee_rate, prev_regime, slippage)
            trades.extend(liq_trades)

            # ─── 5. PRUNING (5 methods, both sides) ──────────
            grid_spacing = calculate_dynamic_spacing(
                prev_atr, spacing_k, spacing_floor, cur_close)
            grid_profit_potential = grid_spacing * order_pct * self.wallet_balance / cur_close

            prune_cooldown = self.config.get('prune_cooldown_bars', 4)

            for pos, acc_profit_attr, last_prune_attr in [
                (self.pos_long,  'accumulated_profit_long',  '_last_prune_bar_long'),
                (self.pos_short, 'accumulated_profit_short', '_last_prune_bar_short'),
            ]:
                if not pos.is_open:
                    continue
                anchor = self.grid_anchor_long if pos.side == 1 else self.grid_anchor_short
                bars_since_prune = i - getattr(self, last_prune_attr)
                on_cooldown = bars_since_prune < prune_cooldown
                acc_profit = getattr(self, acc_profit_attr)
                prune_idx, prune_label = run_pruning_cycle(
                    pos, cur_close, cur_time, prev_kama, prev_atr,
                    anchor, grid_spacing, grid_profit_potential,
                    acc_profit, self.config,
                    is_on_cooldown=on_cooldown)
                if prune_idx >= 0:
                    # Apply slippage to prune market limit
                    exit_price = cur_close
                    if pos.side == 1:  # Long close (sell)
                        exit_price = cur_close * (1 - slippage)
                    else:              # Short close (buy)
                        exit_price = cur_close * (1 + slippage)

                    qty_pruned = pos.fills[prune_idx]['qty'] if prune_idx < len(pos.fills) else 0
                    fee = abs(qty_pruned) * exit_price * fee_rate
                    pnl = pos.close_specific_fill(prune_idx, exit_price, fee)
                    self.wallet_balance += pnl
                    self.metrics['prune_count'] += 1
                    self.metrics['prune_types'][prune_label] = \
                        self.metrics['prune_types'].get(prune_label, 0) + 1
                    setattr(self, last_prune_attr, i)  # record cooldown timestamp

                    # Update profit buffer: consume on offset prune, decrement on any loss
                    if prune_label == LABEL_PRUNE_OFFSET:
                        # Offset prune: pnl is negative → subtracts from buffer
                        setattr(self, acc_profit_attr,
                                max(0.0, acc_profit + pnl))
                    elif pnl < 0:
                        # Non-offset loss (deviance, gap, oldest, funding):
                        # also decrement buffer so it reflects net earned capital
                        setattr(self, acc_profit_attr,
                                max(0.0, acc_profit + pnl))

                    trades.append(self._trade_record(
                        i, cur_time, exit_price, qty_pruned,
                        prune_label, prev_regime, pnl=pnl))

            # ─── 6. TRAILING UP ───────────────────────────────
            if trailing_enabled and prev_regime in (REGIME_UPTREND, REGIME_BREAKOUT_UP):
                if prev_er > trailing_er:
                    top_level = self.grid_anchor_long + grid_spacing * grid_levels
                    if cur_close > top_level:
                        shift = cur_close - self.grid_anchor_long
                        self.grid_anchor_long += shift * 0.5  # Partial shift (conservative)
                        self.trailing_anchor = self.grid_anchor_long
                        self.grid_needs_regen = True
                        self.metrics['trailing_shifts'] += 1

            # ─── 7. VaR CONSTRAINT ────────────────────────────
            total_exposure = (self.pos_long.size + self.pos_short.size) * cur_close
            var_blocked = check_var_constraint(
                total_exposure, prev_vol, max_dd_pct,
                self._total_equity(cur_close))
            if var_blocked:
                self.metrics['var_blocks'] += 1

            # ─── VaR Pre-emptive De-leverage (#6) ─────────────
            # When unrealized drawdown hits 75% of hard cap, force-prune
            # the biggest loser on each side before we breach the limit.
            current_equity = self._total_equity(cur_close)
            drawdown_pct = (self.initial_capital - current_equity) / self.initial_capital
            if drawdown_pct > max_dd_pct * 0.75:
                for pos, side_label in [
                    (self.pos_long,  'accumulated_profit_long'),
                    (self.pos_short, 'accumulated_profit_short'),
                ]:
                    if not pos.fills:
                        continue
                    # Find fill with worst unrealized loss
                    worst_idx = -1
                    worst_loss = 0.0
                    for fi, fill in enumerate(pos.fills):
                        if pos.side == 1:
                            loss = (fill['price'] - cur_close) * fill['qty']
                        else:
                            loss = (cur_close - fill['price']) * fill['qty']
                        if loss > worst_loss:
                            worst_loss = loss
                            worst_idx = fi
                    if worst_idx >= 0:
                        ep = cur_close * (1 - slippage if pos.side == 1 else 1 + slippage)
                        fee = pos.fills[worst_idx]['qty'] * ep * fee_taker
                        pnl = pos.close_specific_fill(worst_idx, ep, fee)
                        self.wallet_balance += pnl
                        self.metrics['prune_count'] += 1
                        self.metrics['prune_types'][LABEL_PRUNE_VAR_WARNING] = \
                            self.metrics['prune_types'].get(LABEL_PRUNE_VAR_WARNING, 0) + 1
                        trades.append(self._trade_record(
                            i, cur_time, ep, 0, LABEL_PRUNE_VAR_WARNING, prev_regime, pnl=pnl))

            # ─── 8. GENERATE GRID + PLACE ORDERS ─────────────
            if self.grid_needs_regen and not var_blocked:
                # Only cancel entry orders — preserve reduce_only TPs that are close to fill
                self.order_book.cancel_all(reduce_only=False)

                # ─── Kelly size multiplier (#5) ───────────────
                kelly_mult = compute_kelly_fraction(
                    self._kelly_trades, prev_regime)
                effective_order_pct = order_pct * kelly_mult

                # ─── Stop-loss de-scaling (#2) ────────────────
                if i < self._descale_until_bar:
                    effective_order_pct *= 0.5

                # ─── Weekend/low-volume de-scaling (#7) ───────
                low_vol_threshold = self.config.get('low_volume_threshold', 0.5)
                if vol_sma7[i] > 1e-9 and vol_arr[i] < low_vol_threshold * vol_sma7[i]:
                    effective_max_pos_pct = max_position_pct * 0.7  # 30% reduction
                else:
                    effective_max_pos_pct = max_position_pct

                self._generate_and_place_grid(
                    cur_close, prev_atr, prev_vol, prev_regime, prev_er, prev_funding,
                    cur_time, grid_levels, spacing_k, spacing_floor,
                    gamma, kappa, effective_order_pct, leverage, max_inv_per_side,
                    effective_max_pos_pct, allow_short, as_time_horizon)
                self.grid_needs_regen = False
                self._prev_regime = prev_regime

            # ─── 9. CHECK FILLS ───────────────────────────────
            filled_orders = self.order_book.check_fills(cur_high, cur_low, fill_prob)
            entry_fills_this_bar = 0
            for order in filled_orders:
                fill_trades = self._process_fill(
                    order, cur_close, cur_time, i, fee_maker, fee_taker, prev_regime)
                trades.extend(fill_trades)

                # Count entry fills (not TPs) for cascade detection
                if not order.get('reduce_only', False):
                    entry_fills_this_bar += 1

                # Track profit for offset pruning (only realised closes produce pnl)
                for t in fill_trades:
                    if t.get('pnl', 0) > 0:
                        if order['direction'] == DIR_LONG:
                            self.accumulated_profit_long += t['pnl']
                        else:
                            self.accumulated_profit_short += t['pnl']

            # Smart grid regen: only regenerate when price moves > regen_drift_mult
            # spacings from anchor OR regime changed. Higher mult = less churn.
            if filled_orders:
                regen_spacing = self._last_grid_spacing if self._last_grid_spacing > 0 else grid_spacing
                regen_drift_mult = self.config.get('regen_drift_mult', 2.0)
                anchor_drift_long = abs(cur_close - self.grid_anchor_long)
                anchor_drift_short = abs(cur_close - self.grid_anchor_short)
                regime_changed = (prev_regime != self._prev_regime)
                if (anchor_drift_long > regen_spacing * regen_drift_mult
                        or anchor_drift_short > regen_spacing * regen_drift_mult
                        or regime_changed):
                    self.grid_needs_regen = True
                if entry_fills_this_bar >= 2:
                    self.grid_needs_regen = True
                self._prev_regime = prev_regime

                # ─── Kelly trade tracking (#5) ────────────────
                for order in filled_orders:
                    for t in trades:
                        if t.get('pnl', 0) != 0:
                            self._kelly_trades.append({
                                'pnl': t['pnl'],
                                'regime': prev_regime,
                            })
                # Keep rolling window
                if len(self._kelly_trades) > self._KELLY_WINDOW:
                    self._kelly_trades = self._kelly_trades[-self._KELLY_WINDOW:]

            # ─── 10. FUNDING RATE ─────────────────────────────
            # The DataFrame provides the 8h funding rate forward-filled to every 15m bar
            # We apply 1/32nd of that rate continuously on every bar
            fr_per_bar = prev_funding / 32.0
            
            if self.pos_long.is_open:
                f_pnl = apply_funding(self.pos_long.size, cur_close, fr_per_bar, 1)
                self.pos_long.add_funding(f_pnl)
                self.wallet_balance += f_pnl
                self.metrics['funding_pnl'] += f_pnl

            if self.pos_short.is_open:
                f_pnl = apply_funding(self.pos_short.size, cur_close, fr_per_bar, -1)
                self.pos_short.add_funding(f_pnl)
                self.wallet_balance += f_pnl
                self.metrics['funding_pnl'] += f_pnl

            # ─── 11. LOG EQUITY ───────────────────────────────
            self._update_equity(cur_close)
            equity_curve[i] = self._total_equity(cur_close)

        # ─── FINAL: Close all positions at last close ─────────
        final_price = data['close'][-1]
        final_time = data['timestamp'][-1]
        if self.pos_long.is_open:
            fee = self.pos_long.size * final_price * fee_rate
            pnl = self.pos_long.close_all(final_price, fee)
            self.wallet_balance += pnl
        if self.pos_short.is_open:
            fee = self.pos_short.size * final_price * fee_rate
            pnl = self.pos_short.close_all(final_price, fee)
            self.wallet_balance += pnl

        equity_curve[-1] = self.wallet_balance

        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'regime_log': regime_log,
            'metrics': self._compute_final_metrics(equity_curve, trades, data),
        }

    # ─── GRID GENERATION ──────────────────────────────────────────

    def _generate_and_place_grid(self, price, atr, vol, regime, er, funding_rate,
                                 timestamp, grid_levels, spacing_k, spacing_floor,
                                 gamma, kappa, order_pct, leverage,
                                 max_inv_per_side, max_position_pct, allow_short,
                                 as_time_horizon=96.0):
        """Generate and place grid orders based on regime and inventory."""

        # Adaptive floor: scales with coin vol ratio (ETH/SOL get wider floors)
        vol_scale = self.config.get('adaptive_floor_scale', 1.5)
        adaptive_floor = calculate_adaptive_floor(atr, price, spacing_floor, vol_scale)
        # Dynamic spacing uses the adaptive floor (in price units → convert to fraction)
        adaptive_floor_pct = adaptive_floor / price if price > 1e-9 else spacing_floor
        base_spacing = calculate_dynamic_spacing(atr, spacing_k, adaptive_floor_pct, price)

        # ─── Asymmetric Grid Spacing based on Regime & ER ────────
        # In a strong uptrend (high ER), longs should be easier to enter (tighter),
        # and shorts should be harder to enter (much wider).
        long_spacing_mult = 1.0
        short_spacing_mult = 1.0

        if regime in (REGIME_UPTREND, REGIME_BREAKOUT_UP):
            long_spacing_mult = max(0.5, 1.0 - (er / 2.0))
            short_spacing_mult = 1.0 + er
        elif regime in (REGIME_DOWNTREND, REGIME_BREAKOUT_DOWN):
            long_spacing_mult = 1.0 + er
            short_spacing_mult = max(0.5, 1.0 - (er / 2.0))

        base_long_spacing = base_spacing * long_spacing_mult
        base_short_spacing = base_spacing * short_spacing_mult

        # Funding bias in NOISE
        long_bias, short_bias = should_bias_for_funding(regime, funding_rate)

        # ─── LONG GRID ────────────────────────────────────
        can_open_long = regime in (REGIME_NOISE, REGIME_UPTREND,
                                    REGIME_BREAKOUT_UP, REGIME_BREAKOUT_DOWN)
        # In downtrend: no new longs, but keep take-profit sells
        can_open_long_entry = regime not in (REGIME_DOWNTREND, REGIME_BREAKOUT_DOWN)

        inv_q_long = normalize_inventory(self.pos_long.size,
                                          max_inv_per_side * order_pct * self.wallet_balance / max(price, 1))
        buy_sp, sell_sp, anchor_long = get_skewed_grid_params(
            price, inv_q_long, gamma, vol, kappa, base_long_spacing, base_long_spacing,
            time_horizon=as_time_horizon)
        self._last_grid_spacing = base_long_spacing

        self.grid_anchor_long = anchor_long

        # ─── Grid level generation: geometric (#4) or arithmetic ────
        grid_mode = self.config.get('grid_mode', 'geometric')
        if grid_mode == 'geometric':
            # buy/sell spacing as fraction of price
            buy_pct  = (buy_sp * long_bias) / max(price, 1e-9)
            sell_pct = sell_sp / max(price, 1e-9)
            buy_levels, sell_levels = generate_geometric_grid_levels(
                anchor_long, buy_pct, sell_pct, grid_levels)
        else:
            buy_levels, sell_levels = generate_grid_levels(
                anchor_long, buy_sp * long_bias, sell_sp, grid_levels)

        # ─── Funding-aware order sizing (#8) ────────────────────
        # In NOISE with strong positive funding: tilt toward short side
        fund_order_pct_long  = order_pct
        fund_order_pct_short = order_pct
        funding_harvest_threshold = self.config.get('funding_harvest_threshold', 0.0002)
        if regime == REGIME_NOISE and abs(funding_rate) > funding_harvest_threshold:
            tilt = 0.25 * order_pct
            if funding_rate > 0:  # Positive: shorts receive
                fund_order_pct_short = min(order_pct + tilt, order_pct * 1.5)
                fund_order_pct_long  = max(order_pct - tilt, order_pct * 0.5)
            else:  # Negative: longs receive
                fund_order_pct_long  = min(order_pct + tilt, order_pct * 1.5)
                fund_order_pct_short = max(order_pct - tilt, order_pct * 0.5)

        # Place long grid buy orders (entry)
        if can_open_long_entry and self.pos_long.num_fills < max_inv_per_side:
            max_notional = self.wallet_balance * max_position_pct
            current_notional = self.pos_long.size * price
            for lvl_i, lvl_price in enumerate(buy_levels):
                if lvl_price <= 0 or current_notional >= max_notional:
                    break
                qty = calculate_order_qty(self.wallet_balance, lvl_price, fund_order_pct_long, leverage)
                if qty > 1e-12:
                    self.order_book.add_order(
                        SIDE_BUY, lvl_price, qty, timestamp,
                        lvl_i, DIR_LONG)
                    current_notional += qty * lvl_price

        # Place long grid sell orders (take-profit on existing longs)
        # Concentrated: 60% at nearest level, 40% spread across remaining levels
        if self.pos_long.is_open:
            tp_concentration = self.config.get('tp_concentration', 0.6)
            n_levels = max(grid_levels, 1)
            for lvl_i, lvl_price in enumerate(sell_levels):
                if lvl_i == 0:
                    tp_qty = self.pos_long.size * tp_concentration
                else:
                    remaining_levels = max(n_levels - 1, 1)
                    tp_qty = self.pos_long.size * (1.0 - tp_concentration) / remaining_levels
                if tp_qty < 1e-12:
                    break
                self.order_book.add_order(
                    SIDE_SELL, lvl_price, tp_qty, timestamp,
                    lvl_i, DIR_LONG, reduce_only=True)

        # ─── SHORT GRID ───────────────────────────────────
        if allow_short:
            can_open_short_entry = regime not in (REGIME_UPTREND, REGIME_BREAKOUT_UP)

            inv_q_short = normalize_inventory(
                -self.pos_short.size,
                max_inv_per_side * order_pct * self.wallet_balance / max(price, 1))
            buy_sp_s, sell_sp_s, anchor_short = get_skewed_grid_params(
                price, inv_q_short, gamma, vol, kappa, base_short_spacing, base_short_spacing,
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

            # Short grid sell orders (entry)
            if can_open_short_entry and self.pos_short.num_fills < max_inv_per_side:
                max_notional = self.wallet_balance * max_position_pct
                current_notional = self.pos_short.size * price
                for lvl_i, lvl_price in enumerate(sell_levels_s):
                    if current_notional >= max_notional:
                        break
                    qty = calculate_order_qty(self.wallet_balance, lvl_price, fund_order_pct_short, leverage)
                    if qty > 1e-12:
                        self.order_book.add_order(
                            SIDE_SELL, lvl_price, qty, timestamp,
                            lvl_i, DIR_SHORT)
                        current_notional += qty * lvl_price

            # Short grid buy orders (take-profit on existing shorts)
            # Concentrated: 60% at nearest level, 40% spread across remaining levels
            if self.pos_short.is_open:
                tp_concentration = self.config.get('tp_concentration', 0.6)
                n_levels = max(grid_levels, 1)
                for lvl_i, lvl_price in enumerate(buy_levels_s):
                    if lvl_i == 0:
                        tp_qty = self.pos_short.size * tp_concentration
                    else:
                        remaining_levels = max(n_levels - 1, 1)
                        tp_qty = self.pos_short.size * (1.0 - tp_concentration) / remaining_levels
                    if tp_qty < 1e-12 or lvl_price <= 0:
                        break
                    self.order_book.add_order(
                        SIDE_BUY, lvl_price, tp_qty, timestamp,
                        lvl_i, DIR_SHORT, reduce_only=True)

    # ─── FILL PROCESSING ──────────────────────────────────────────

    def _process_fill(self, order, current_price, timestamp, bar_idx,
                      fee_maker, fee_taker, regime) -> list:
        """Process a filled order. Returns list of trade records.

        All order-book fills (entries AND take-profits) are limit orders → maker fee.
        Stops, prunes, and liquidations are market/taker orders and apply fee_taker
        directly in their own code paths, not through here.
        """
        trades = []
        qty = order['qty']
        fill_price = order['price']
        # Both grid entries and TP limit orders are resting limit orders → maker fee
        fee_rate = fee_maker
        fee = qty * fill_price * fee_rate

        if order['direction'] == DIR_LONG:
            if order['side'] == SIDE_BUY and not order.get('reduce_only', False):
                # Opening long
                self.pos_long.add_fill(fill_price, qty, timestamp, fee)
                self.wallet_balance -= fee
                self.metrics['longs_opened'] += 1
                trades.append(self._trade_record(
                    bar_idx, timestamp, fill_price, qty,
                    LABEL_BUY_OPEN_LONG, regime))
            elif order['side'] == SIDE_SELL:
                # Closing long (take-profit)
                pnl = self.pos_long.close_fill(fill_price, qty, fee)
                self.wallet_balance += pnl
                self.metrics['longs_closed'] += 1
                trades.append(self._trade_record(
                    bar_idx, timestamp, fill_price, qty,
                    LABEL_SELL_CLOSE_LONG, regime, pnl=pnl))

        elif order['direction'] == DIR_SHORT:
            if order['side'] == SIDE_SELL and not order.get('reduce_only', False):
                # Opening short
                self.pos_short.add_fill(fill_price, qty, timestamp, fee)
                self.wallet_balance -= fee
                self.metrics['shorts_opened'] += 1
                trades.append(self._trade_record(
                    bar_idx, timestamp, fill_price, qty,
                    LABEL_SELL_OPEN_SHORT, regime))
            elif order['side'] == SIDE_BUY:
                # Closing short (take-profit)
                pnl = self.pos_short.close_fill(fill_price, qty, fee)
                self.wallet_balance += pnl
                self.metrics['shorts_closed'] += 1
                trades.append(self._trade_record(
                    bar_idx, timestamp, fill_price, qty,
                    LABEL_BUY_CLOSE_SHORT, regime, pnl=pnl))

        return trades

    # ─── STOP LOSS ────────────────────────────────────────────────

    def _check_stop_loss(self, price, low, high, timestamp, bar_idx,
                         kama, atr, sl_mult, fee_rate, regime, slippage=0.0) -> list:
        """ATR-based stop loss for both sides.

        Partial stop: if position has >1 fill, first trigger closes 50%
        and sets a wider stop (1.5× mult) for the remainder. Single-fill
        positions close 100% immediately.
        """
        trades = []

        # Long stop
        if self.pos_long.is_open:
            # Use wider multiplier if first stage already fired
            effective_mult = sl_mult * 1.5 if self._partial_stop_fired_long else sl_mult
            stop_price = self.pos_long.avg_entry - effective_mult * atr

            # Stage 1 (#3): anti-wick — only fire partial stop on candle CLOSE
            # Stage 2: use wick (last resort safety net)
            stage1_trigger = price <= stop_price   # 'price' here is cur_close
            stage2_trigger = low <= stop_price

            if (stage1_trigger and self.pos_long.num_fills > 1
                    and not self._partial_stop_fired_long):
                exit_price = stop_price * (1 - slippage)
                close_qty = self.pos_long.size * 0.5
                fee = close_qty * exit_price * fee_rate
                pnl = self.pos_long.close_fill(exit_price, close_qty, fee)
                self.wallet_balance += pnl
                self.metrics['stops_long'] += 1
                self._partial_stop_fired_long = True
                # Stop-loss cooldown tracking (#2)
                if bar_idx - self._last_stop_bar <= self._STOP_COOLDOWN_BARS:
                    self._consecutive_stops += 1
                else:
                    self._consecutive_stops = 1
                self._last_stop_bar = bar_idx
                if self._consecutive_stops >= self._STOP_COOLDOWN_THRESH:
                    self._descale_until_bar = bar_idx + self._STOP_COOLDOWN_BARS
                trades.append(self._trade_record(
                    bar_idx, timestamp, exit_price, close_qty,
                    LABEL_STOP_LONG, regime, pnl=pnl))
            elif stage2_trigger:
                # Stage 2 (or single fill): close everything
                exit_price = stop_price * (1 - slippage)
                fee = self.pos_long.size * exit_price * fee_rate
                pnl = self.pos_long.close_all(exit_price, fee)
                self.wallet_balance += pnl
                self.metrics['stops_long'] += 1
                self._partial_stop_fired_long = False
                if bar_idx - self._last_stop_bar <= self._STOP_COOLDOWN_BARS:
                    self._consecutive_stops += 1
                else:
                    self._consecutive_stops = 1
                self._last_stop_bar = bar_idx
                if self._consecutive_stops >= self._STOP_COOLDOWN_THRESH:
                    self._descale_until_bar = bar_idx + self._STOP_COOLDOWN_BARS
                trades.append(self._trade_record(
                    bar_idx, timestamp, exit_price, 0,
                    LABEL_STOP_LONG, regime, pnl=pnl))
            if stage1_trigger or stage2_trigger:
                self.grid_needs_regen = True

        # Reset partial stop state if position closed elsewhere
        if not self.pos_long.is_open:
            self._partial_stop_fired_long = False

        # Short stop
        if self.pos_short.is_open:
            effective_mult = sl_mult * 1.5 if self._partial_stop_fired_short else sl_mult
            stop_price = self.pos_short.avg_entry + effective_mult * atr

            # Stage 1 (#3): anti-wick — only fire partial stop on candle CLOSE
            stage1_trigger = price >= stop_price   # 'price' is cur_close
            stage2_trigger = high >= stop_price

            if (stage1_trigger and self.pos_short.num_fills > 1
                    and not self._partial_stop_fired_short):
                exit_price = stop_price * (1 + slippage)
                close_qty = self.pos_short.size * 0.5
                fee = close_qty * exit_price * fee_rate
                pnl = self.pos_short.close_fill(exit_price, close_qty, fee)
                self.wallet_balance += pnl
                self.metrics['stops_short'] += 1
                self._partial_stop_fired_short = True
                if bar_idx - self._last_stop_bar <= self._STOP_COOLDOWN_BARS:
                    self._consecutive_stops += 1
                else:
                    self._consecutive_stops = 1
                self._last_stop_bar = bar_idx
                if self._consecutive_stops >= self._STOP_COOLDOWN_THRESH:
                    self._descale_until_bar = bar_idx + self._STOP_COOLDOWN_BARS
                trades.append(self._trade_record(
                    bar_idx, timestamp, exit_price, close_qty,
                    LABEL_STOP_SHORT, regime, pnl=pnl))
            elif stage2_trigger:
                exit_price = stop_price * (1 + slippage)
                fee = self.pos_short.size * exit_price * fee_rate
                pnl = self.pos_short.close_all(exit_price, fee)
                self.wallet_balance += pnl
                self.metrics['stops_short'] += 1
                self._partial_stop_fired_short = False
                if bar_idx - self._last_stop_bar <= self._STOP_COOLDOWN_BARS:
                    self._consecutive_stops += 1
                else:
                    self._consecutive_stops = 1
                self._last_stop_bar = bar_idx
                if self._consecutive_stops >= self._STOP_COOLDOWN_THRESH:
                    self._descale_until_bar = bar_idx + self._STOP_COOLDOWN_BARS
                trades.append(self._trade_record(
                    bar_idx, timestamp, exit_price, 0,
                    LABEL_STOP_SHORT, regime, pnl=pnl))
            if stage1_trigger or stage2_trigger:
                self.grid_needs_regen = True

        # Reset partial stop state if position closed elsewhere
        if not self.pos_short.is_open:
            self._partial_stop_fired_short = False

        return trades

    # ─── LIQUIDATION ──────────────────────────────────────────────

    def _check_liquidation(self, price, timestamp, bar_idx, fee_rate, regime, slippage=0.0) -> list:
        """Check if either position would be liquidated."""
        trades = []
        leverage = self.config.get('leverage', 1.0)

        if self.pos_long.is_open and leverage > 1.0:
            liq_price = calculate_liquidation_price(
                self.pos_long.avg_entry, self.pos_long.size,
                self.wallet_balance, 1, leverage)
            if price <= liq_price and liq_price > 0:
                exit_price = price * (1 - slippage)
                pnl = self.pos_long.close_all(exit_price, 0)
                self.wallet_balance += pnl
                self.metrics['liquidations'] += 1
                trades.append(self._trade_record(
                    bar_idx, timestamp, price, 0,
                    LABEL_LIQUIDATION, regime, pnl=pnl))
                self.grid_needs_regen = True

        if self.pos_short.is_open and leverage > 1.0:
            liq_price = calculate_liquidation_price(
                self.pos_short.avg_entry, self.pos_short.size,
                self.wallet_balance, -1, leverage)
            if price >= liq_price and liq_price > 0:
                exit_price = price * (1 + slippage)
                pnl = self.pos_short.close_all(exit_price, 0)
                self.wallet_balance += pnl
                self.metrics['liquidations'] += 1
                trades.append(self._trade_record(
                    bar_idx, timestamp, price, 0,
                    LABEL_LIQUIDATION, regime, pnl=pnl))
                self.grid_needs_regen = True

        return trades

    # ─── EQUITY CALCULATION ───────────────────────────────────────

    def _update_equity(self, price):
        """Update unrealized PnL for both positions."""
        self.pos_long.update_unrealized(price)
        self.pos_short.update_unrealized(price)

    def _total_equity(self, price) -> float:
        """Wallet balance + unrealized PnL on both sides."""
        self._update_equity(price)
        return (self.wallet_balance
                + self.pos_long.unrealized_pnl
                + self.pos_short.unrealized_pnl)

    # ─── TRADE RECORDS ────────────────────────────────────────────

    @staticmethod
    def _trade_record(bar_idx, timestamp, price, qty, label, regime,
                      pnl=0.0) -> dict:
        return {
            'bar': bar_idx,
            'timestamp': timestamp,
            'price': price,
            'qty': qty,
            'label': label,
            'regime': regime,
            'pnl': pnl,
        }

    # ─── FINAL METRICS ────────────────────────────────────────────

    def _compute_final_metrics(self, equity_curve, trades, data) -> dict:
        """Compute comprehensive performance metrics."""
        ec = equity_curve
        n = len(ec)

        # Basic returns
        total_return_pct = (ec[-1] - ec[0]) / ec[0] * 100 if ec[0] > 0 else 0

        # Buy & hold
        bh_return = (data['close'][-1] - data['close'][0]) / data['close'][0] * 100

        # Max drawdown
        peak = ec[0]
        max_dd = 0.0
        for i in range(1, n):
            if ec[i] > peak:
                peak = ec[i]
            dd = (peak - ec[i]) / peak
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = max_dd * 100

        # Returns for Sharpe/Sortino
        returns = np.diff(ec) / ec[:-1]
        returns = returns[np.isfinite(returns)]

        mean_ret = np.mean(returns) if len(returns) > 0 else 0
        std_ret = np.std(returns) if len(returns) > 0 else 1e-9

        # Sharpe (annualized for 15m: sqrt(96*365))
        ann_factor = np.sqrt(96 * 365)
        sharpe = (mean_ret / max(std_ret, 1e-9)) * ann_factor

        # Sortino (downside deviation only)
        neg_returns = returns[returns < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 1e-9
        sortino = (mean_ret / max(downside_std, 1e-9)) * ann_factor

        # Calmar
        calmar = (total_return_pct / max(max_dd_pct, 1e-9))

        # Win rate & profit factor
        pnls = [t['pnl'] for t in trades if t['pnl'] != 0]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / max(len(pnls), 1) * 100
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        metrics = {
            'total_return_pct': round(total_return_pct, 2),
            'buy_hold_return_pct': round(bh_return, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'calmar_ratio': round(calmar, 3),
            'win_rate_pct': round(win_rate, 1),
            'profit_factor': round(profit_factor, 3),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'total_trades': len(pnls),
            'final_capital': round(ec[-1], 2),
            'funding_pnl': round(self.metrics['funding_pnl'], 2),
        }
        metrics.update(self.metrics)
        return metrics
