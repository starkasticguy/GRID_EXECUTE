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
    REGIME_NOISE, REGIME_UPTREND, REGIME_DOWNTREND,
    REGIME_BREAKOUT_UP, REGIME_BREAKOUT_DOWN, REGIME_NAMES,
)
from core.atr import calculate_atr, calculate_z_score, calculate_rolling_volatility
from core.grid import generate_grid_levels, calculate_order_qty, calculate_dynamic_spacing
from core.inventory import get_skewed_grid_params, normalize_inventory
from core.risk import (
    calculate_liquidation_price, calculate_var_95, check_var_constraint,
    calculate_unrealized_pnl, calculate_funding_pnl,
)
from core.funding import (
    generate_synthetic_funding_rates, is_funding_interval,
    apply_funding, should_bias_for_funding,
)
from engine.types import (
    PositionTracker, SIDE_BUY, SIDE_SELL, DIR_LONG, DIR_SHORT,
    LABEL_BUY_OPEN_LONG, LABEL_SELL_CLOSE_LONG,
    LABEL_SELL_OPEN_SHORT, LABEL_BUY_CLOSE_SHORT,
    LABEL_STOP_LONG, LABEL_STOP_SHORT,
    LABEL_CIRCUIT_BREAKER, LABEL_LIQUIDATION,
)
from engine.matching import OrderBook
from engine.pruning import run_pruning_cycle
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

        # Trailing state
        self.trailing_anchor = 0.0

        # Circuit breaker state
        self.halted = False

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

    def prepare_indicators(self, df) -> dict:
        """
        Pre-compute all indicators from OHLCV DataFrame.

        Uses prev-bar indicators for decisions (no lookahead).
        """
        closes = df['close'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        opens = df['open'].values.astype(np.float64)
        timestamps = df['timestamp'].values.astype(np.float64)

        kama_p = self.config['kama_period']
        atr_p = self.config['atr_period']

        er = calculate_er(closes, kama_p)
        kama = calculate_kama(closes, er, self.config['kama_fast'],
                              self.config['kama_slow'])
        atr = calculate_atr(highs, lows, closes, atr_p)
        regime = detect_regime(kama, er, atr,
                               self.config['regime_threshold'],
                               self.config['er_trend_threshold'])
        z_score = calculate_z_score(closes, atr, 20)
        volatility = calculate_rolling_volatility(closes, atr_p)
        funding_rates = generate_synthetic_funding_rates(closes)

        return {
            'open': opens, 'high': highs, 'low': lows, 'close': closes,
            'timestamp': timestamps,
            'kama': kama, 'er': er, 'atr': atr,
            'regime': regime, 'z_score': z_score,
            'volatility': volatility, 'funding_rates': funding_rates,
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

        # Track accumulated grid profit for profit-offset pruning
        accumulated_profit_long = 0.0
        accumulated_profit_short = 0.0

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

            for pos, acc_profit in [
                (self.pos_long, accumulated_profit_long),
                (self.pos_short, accumulated_profit_short),
            ]:
                if not pos.is_open:
                    continue
                anchor = self.grid_anchor_long if pos.side == 1 else self.grid_anchor_short
                prune_idx, prune_label = run_pruning_cycle(
                    pos, cur_close, cur_time, prev_kama, prev_atr,
                    anchor, grid_spacing, grid_profit_potential,
                    acc_profit, self.config)
                if prune_idx >= 0:
                    # Apply slippage to prune market limit
                    exit_price = cur_close
                    if pos.side == 1:  # Long close (sell)
                        exit_price = cur_close * (1 - slippage)
                    else:              # Short close (buy)
                        exit_price = cur_close * (1 + slippage)

                    fee = abs(pos.fills[prune_idx]['qty']) * exit_price * fee_rate
                    pnl = pos.close_specific_fill(prune_idx, exit_price, fee)
                    self.wallet_balance += pnl
                    self.metrics['prune_count'] += 1
                    self.metrics['prune_types'][prune_label] = \
                        self.metrics['prune_types'].get(prune_label, 0) + 1
                    trades.append(self._trade_record(
                        i, cur_time, exit_price,
                        pos.fills[prune_idx]['qty'] if prune_idx < len(pos.fills) else 0,
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

            # ─── 8. GENERATE GRID + PLACE ORDERS ─────────────
            if self.grid_needs_regen and not var_blocked:
                self.order_book.cancel_all()
                self._generate_and_place_grid(
                    cur_close, prev_atr, prev_vol, prev_regime, prev_funding,
                    cur_time, grid_levels, spacing_k, spacing_floor,
                    gamma, kappa, order_pct, leverage, max_inv_per_side,
                    max_position_pct, allow_short, as_time_horizon)
                self.grid_needs_regen = False
                self._prev_regime = prev_regime

            # ─── 9. CHECK FILLS ───────────────────────────────
            filled_orders = self.order_book.check_fills(cur_high, cur_low, fill_prob)
            for order in filled_orders:
                fill_trades = self._process_fill(
                    order, cur_close, cur_time, i, fee_maker, fee_taker, prev_regime)
                trades.extend(fill_trades)

                # Track profit for offset pruning
                for t in fill_trades:
                    if t.get('pnl', 0) > 0:
                        if order['direction'] == DIR_LONG:
                            accumulated_profit_long += t['pnl']
                        else:
                            accumulated_profit_short += t['pnl']

            # Smart grid regen: only regenerate when price moves >1 spacing
            # from anchor or regime changed. Avoids cancel-and-replace churn
            # that loses queue position on every fill.
            if filled_orders:
                regen_spacing = self._last_grid_spacing if self._last_grid_spacing > 0 else grid_spacing
                anchor_drift_long = abs(cur_close - self.grid_anchor_long)
                anchor_drift_short = abs(cur_close - self.grid_anchor_short)
                regime_changed = (prev_regime != self._prev_regime)
                if anchor_drift_long > regen_spacing or anchor_drift_short > regen_spacing or regime_changed:
                    self.grid_needs_regen = True
                self._prev_regime = prev_regime

            # ─── 10. FUNDING RATE ─────────────────────────────
            if is_funding_interval(i):
                fr = prev_funding
                if self.pos_long.is_open:
                    f_pnl = apply_funding(self.pos_long.size, cur_close, fr, 1)
                    self.pos_long.add_funding(f_pnl)
                    self.wallet_balance += f_pnl
                    self.metrics['funding_pnl'] += f_pnl

                if self.pos_short.is_open:
                    f_pnl = apply_funding(self.pos_short.size, cur_close, fr, -1)
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

    def _generate_and_place_grid(self, price, atr, vol, regime, funding_rate,
                                 timestamp, grid_levels, spacing_k, spacing_floor,
                                 gamma, kappa, order_pct, leverage,
                                 max_inv_per_side, max_position_pct, allow_short,
                                 as_time_horizon=96.0):
        """Generate and place grid orders based on regime and inventory."""

        base_spacing = calculate_dynamic_spacing(atr, spacing_k, spacing_floor, price)

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
            price, inv_q_long, gamma, vol, kappa, base_spacing, base_spacing,
            time_horizon=as_time_horizon)
        self._last_grid_spacing = base_spacing

        self.grid_anchor_long = anchor_long
        buy_levels, sell_levels = generate_grid_levels(
            anchor_long, buy_sp * long_bias, sell_sp, grid_levels)

        # Place long grid buy orders (entry)
        if can_open_long_entry and self.pos_long.num_fills < max_inv_per_side:
            max_notional = self.wallet_balance * max_position_pct
            current_notional = self.pos_long.size * price
            for lvl_i, lvl_price in enumerate(buy_levels):
                if lvl_price <= 0 or current_notional >= max_notional:
                    break
                qty = calculate_order_qty(self.wallet_balance, lvl_price, order_pct, leverage)
                if qty > 1e-12:
                    self.order_book.add_order(
                        SIDE_BUY, lvl_price, qty, timestamp,
                        lvl_i, DIR_LONG)
                    current_notional += qty * lvl_price

        # Place long grid sell orders (take-profit on existing longs)
        if self.pos_long.is_open:
            tp_qty_per_level = self.pos_long.size / max(grid_levels, 1)
            for lvl_i, lvl_price in enumerate(sell_levels):
                if tp_qty_per_level < 1e-12:
                    break
                self.order_book.add_order(
                    SIDE_SELL, lvl_price, tp_qty_per_level, timestamp,
                    lvl_i, DIR_LONG, reduce_only=True)

        # ─── SHORT GRID ───────────────────────────────────
        if allow_short:
            can_open_short_entry = regime not in (REGIME_UPTREND, REGIME_BREAKOUT_UP)

            inv_q_short = normalize_inventory(
                -self.pos_short.size,
                max_inv_per_side * order_pct * self.wallet_balance / max(price, 1))
            buy_sp_s, sell_sp_s, anchor_short = get_skewed_grid_params(
                price, inv_q_short, gamma, vol, kappa, base_spacing, base_spacing,
                time_horizon=as_time_horizon)

            self.grid_anchor_short = anchor_short
            buy_levels_s, sell_levels_s = generate_grid_levels(
                anchor_short, buy_sp_s, sell_sp_s * short_bias, grid_levels)

            # Short grid sell orders (entry)
            if can_open_short_entry and self.pos_short.num_fills < max_inv_per_side:
                max_notional = self.wallet_balance * max_position_pct
                current_notional = self.pos_short.size * price
                for lvl_i, lvl_price in enumerate(sell_levels_s):
                    if current_notional >= max_notional:
                        break
                    qty = calculate_order_qty(self.wallet_balance, lvl_price, order_pct, leverage)
                    if qty > 1e-12:
                        self.order_book.add_order(
                            SIDE_SELL, lvl_price, qty, timestamp,
                            lvl_i, DIR_SHORT)
                        current_notional += qty * lvl_price

            # Short grid buy orders (take-profit on existing shorts)
            if self.pos_short.is_open:
                tp_qty_per_level = self.pos_short.size / max(grid_levels, 1)
                for lvl_i, lvl_price in enumerate(buy_levels_s):
                    if tp_qty_per_level < 1e-12 or lvl_price <= 0:
                        break
                    self.order_book.add_order(
                        SIDE_BUY, lvl_price, tp_qty_per_level, timestamp,
                        lvl_i, DIR_SHORT, reduce_only=True)

    # ─── FILL PROCESSING ──────────────────────────────────────────

    def _process_fill(self, order, current_price, timestamp, bar_idx,
                      fee_maker, fee_taker, regime) -> list:
        """Process a filled order. Returns list of trade records.

        Limit grid entries pay maker fee; reduce_only closes (TPs) pay taker fee.
        This matches live Binance behaviour where entries are LIMIT and
        take-profits are TAKE_PROFIT_MARKET (taker).
        """
        trades = []
        qty = order['qty']
        fill_price = order['price']
        # Entries are limit orders (maker); closes are market/TP orders (taker)
        is_entry = not order.get('reduce_only', False)
        fee_rate = fee_maker if is_entry else fee_taker
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
            if low <= stop_price:
                exit_price = stop_price * (1 - slippage)

                if self.pos_long.num_fills > 1 and not self._partial_stop_fired_long:
                    # Stage 1: close 50% of position
                    close_qty = self.pos_long.size * 0.5
                    fee = close_qty * exit_price * fee_rate
                    pnl = self.pos_long.close_fill(exit_price, close_qty, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_long'] += 1
                    self._partial_stop_fired_long = True
                    trades.append(self._trade_record(
                        bar_idx, timestamp, exit_price, close_qty,
                        LABEL_STOP_LONG, regime, pnl=pnl))
                else:
                    # Stage 2 (or single fill): close everything
                    fee = self.pos_long.size * exit_price * fee_rate
                    pnl = self.pos_long.close_all(exit_price, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_long'] += 1
                    self._partial_stop_fired_long = False
                    trades.append(self._trade_record(
                        bar_idx, timestamp, exit_price, 0,
                        LABEL_STOP_LONG, regime, pnl=pnl))
                self.grid_needs_regen = True

        # Reset partial stop state if position closed elsewhere
        if not self.pos_long.is_open:
            self._partial_stop_fired_long = False

        # Short stop
        if self.pos_short.is_open:
            effective_mult = sl_mult * 1.5 if self._partial_stop_fired_short else sl_mult
            stop_price = self.pos_short.avg_entry + effective_mult * atr
            if high >= stop_price:
                exit_price = stop_price * (1 + slippage)

                if self.pos_short.num_fills > 1 and not self._partial_stop_fired_short:
                    # Stage 1: close 50% of position
                    close_qty = self.pos_short.size * 0.5
                    fee = close_qty * exit_price * fee_rate
                    pnl = self.pos_short.close_fill(exit_price, close_qty, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_short'] += 1
                    self._partial_stop_fired_short = True
                    trades.append(self._trade_record(
                        bar_idx, timestamp, exit_price, close_qty,
                        LABEL_STOP_SHORT, regime, pnl=pnl))
                else:
                    # Stage 2 (or single fill): close everything
                    fee = self.pos_short.size * exit_price * fee_rate
                    pnl = self.pos_short.close_all(exit_price, fee)
                    self.wallet_balance += pnl
                    self.metrics['stops_short'] += 1
                    self._partial_stop_fired_short = False
                    trades.append(self._trade_record(
                        bar_idx, timestamp, exit_price, 0,
                        LABEL_STOP_SHORT, regime, pnl=pnl))
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
