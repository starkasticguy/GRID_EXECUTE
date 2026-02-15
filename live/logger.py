"""
TradeLogger — Structured logging with per-trade performance metrics.

Provides:
  - Console output (formatted)
  - File logging with rotation
  - Per-trade metrics (PnL, return %, hold time, R-multiple)
  - Running aggregate metrics (win rate, profit factor, Sharpe estimate)
"""
import os
import logging
import logging.handlers
import math
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from core.kama import REGIME_NAMES


class TradeLogger:

    def __init__(self, log_dir: str, symbol: str, log_level: str = 'INFO'):
        self.symbol = symbol
        os.makedirs(log_dir, exist_ok=True)

        # Running stats
        self._trade_pnls = []
        self._trade_returns = []
        self._wins = 0
        self._losses = 0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._total_trades = 0
        self._hold_hours = []
        self._session_start = datetime.now(timezone.utc)

        # Entry tracking for hold-time and R-multiple calculation
        self._open_entries = {}  # label -> list of {price, qty, timestamp}

        # Set up loggers
        self._log = logging.getLogger('trade_logger')
        self._log.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Avoid duplicate handlers on re-init
        if not self._log.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            ch.setFormatter(logging.Formatter('%(message)s'))
            self._log.addHandler(ch)

            # File handler with rotation (10MB, 5 backups)
            log_file = os.path.join(log_dir, f'{symbol}.log')
            fh = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            self._log.addHandler(fh)

    def log_trade(self, trade: dict):
        """Log a trade event with per-trade metrics."""
        ts = trade.get('timestamp', 0)
        dt = _ms_to_str(ts)
        label = trade.get('label', '?')
        price = trade.get('price', 0)
        qty = trade.get('qty', 0)
        pnl = trade.get('pnl', 0)
        regime = trade.get('regime', 0)
        regime_name = REGIME_NAMES.get(regime, 'UNKNOWN')

        # Track entry for hold-time calc
        if 'OPEN' in label:
            self._track_entry(label, price, qty, ts)
            self._log.info(
                f"[{dt}] {label} | {self.symbol}\n"
                f"  Price: ${price:,.2f} | Qty: {qty:.6f} | Regime: {regime_name}")
            return

        # For close/stop/prune — compute metrics
        hold_hours = self._calc_hold_hours(label, ts)
        r_mult = self._calc_r_multiple(pnl, price, qty)

        self._update_running_stats(pnl, hold_hours)

        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        ret_pct = (pnl / max(price * qty, 1e-9)) * 100 if qty > 0 else 0

        lines = [f"[{dt}] {label} | {self.symbol}"]
        lines.append(
            f"  Price: ${price:,.2f} | Qty: {qty:.6f} | PnL: {pnl_str} ({ret_pct:+.2f}%)")
        if hold_hours is not None:
            lines.append(
                f"  Hold: {hold_hours:.1f}h | R: {r_mult:+.2f} | Regime: {regime_name}")
        else:
            lines.append(f"  Regime: {regime_name}")

        self._log.info('\n'.join(lines))

    def log_equity(self, timestamp: int, wallet_balance: float,
                   unrealized_long: float, unrealized_short: float,
                   pos_long_size: float, pos_short_size: float,
                   regime: int):
        """Log equity snapshot (every bar)."""
        dt = _ms_to_str(timestamp)
        total = wallet_balance + unrealized_long + unrealized_short
        regime_name = REGIME_NAMES.get(regime, '?')

        self._log.info(
            f"[{dt}] EQUITY | ${total:,.2f} | "
            f"Wallet: ${wallet_balance:,.2f} | "
            f"UPnL_L: ${unrealized_long:+,.2f} | UPnL_S: ${unrealized_short:+,.2f} | "
            f"Pos_L: {pos_long_size:.6f} | Pos_S: {pos_short_size:.6f} | "
            f"Regime: {regime_name}")

    def log_event(self, event_type: str, details: str):
        """Log a non-trade event."""
        dt = _ms_to_str(int(datetime.now(timezone.utc).timestamp() * 1000))
        self._log.info(f"[{dt}] EVENT:{event_type} | {details}")

    def log_grid_placement(self, direction: str, num_orders: int,
                            price_range: tuple):
        """Log grid order placement summary."""
        dt = _ms_to_str(int(datetime.now(timezone.utc).timestamp() * 1000))
        low, high = price_range
        self._log.info(
            f"[{dt}] GRID:{direction} | {num_orders} orders | "
            f"Range: ${low:,.2f} - ${high:,.2f}")

    def get_running_metrics(self) -> dict:
        """Compute running performance metrics."""
        total = self._total_trades
        if total == 0:
            return {
                'total_trades': 0, 'win_rate_pct': 0.0, 'profit_factor': 0.0,
                'avg_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'best_trade': 0.0, 'worst_trade': 0.0, 'total_pnl': 0.0,
                'avg_hold_hours': 0.0, 'estimated_sharpe': 0.0,
            }

        total_pnl = self._gross_profit - self._gross_loss
        pnls = self._trade_pnls

        # Estimated Sharpe from trade PnLs
        if len(pnls) > 1:
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            # Annualize: assume ~6.5 trades/day (rough grid estimate)
            est_sharpe = (mean_pnl / max(std_pnl, 1e-9)) * math.sqrt(365 * 6.5)
        else:
            est_sharpe = 0.0

        return {
            'total_trades': total,
            'win_rate_pct': round(self._wins / max(total, 1) * 100, 1),
            'profit_factor': round(
                self._gross_profit / max(self._gross_loss, 1e-9), 3),
            'avg_pnl': round(total_pnl / total, 2),
            'avg_win': round(
                self._gross_profit / max(self._wins, 1), 2),
            'avg_loss': round(
                self._gross_loss / max(self._losses, 1), 2),
            'best_trade': round(max(pnls) if pnls else 0, 2),
            'worst_trade': round(min(pnls) if pnls else 0, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_hold_hours': round(
                np.mean(self._hold_hours) if self._hold_hours else 0, 1),
            'estimated_sharpe': round(est_sharpe, 3),
        }

    def log_session_summary(self, metrics: dict, session_duration_hours: float):
        """Log formatted session summary on shutdown."""
        m = self.get_running_metrics()
        m.update(metrics)

        border = '=' * 62
        self._log.info(f"\n{border}")
        self._log.info(f"  SESSION SUMMARY — {self.symbol}")
        self._log.info(f"  Duration: {session_duration_hours:.1f} hours")
        self._log.info(f"{border}")
        self._log.info(f"  Total Trades:    {m.get('total_trades', 0)}")
        self._log.info(f"  Win Rate:        {m.get('win_rate_pct', 0):.1f}%")
        self._log.info(f"  Profit Factor:   {m.get('profit_factor', 0):.3f}")
        self._log.info(f"  Total PnL:       ${m.get('total_pnl', 0):+,.2f}")
        self._log.info(f"  Avg PnL/Trade:   ${m.get('avg_pnl', 0):+,.2f}")
        self._log.info(f"  Avg Hold:        {m.get('avg_hold_hours', 0):.1f}h")
        self._log.info(f"  Est. Sharpe:     {m.get('estimated_sharpe', 0):.3f}")
        self._log.info(f"  Longs Opened:    {m.get('longs_opened', 0)}")
        self._log.info(f"  Shorts Opened:   {m.get('shorts_opened', 0)}")
        self._log.info(f"  Stops (L/S):     {m.get('stops_long', 0)}/{m.get('stops_short', 0)}")
        self._log.info(f"  Prunes:          {m.get('prune_count', 0)}")
        self._log.info(f"  CB Triggers:     {m.get('circuit_breaker_triggers', 0)}")
        self._log.info(f"  Trailing Shifts: {m.get('trailing_shifts', 0)}")
        self._log.info(f"  Funding PnL:     ${m.get('funding_pnl', 0):+,.2f}")
        self._log.info(border)

    # ─── Internal Helpers ────────────────────────────────────────

    def _track_entry(self, label: str, price: float, qty: float, ts: int):
        """Track an entry for later hold-time calculation."""
        # Map entry labels to their close labels
        if label not in self._open_entries:
            self._open_entries[label] = []
        self._open_entries[label].append({
            'price': price, 'qty': qty, 'timestamp': ts})

    def _calc_hold_hours(self, close_label: str, close_ts: int) -> Optional[float]:
        """Calculate hold time for a closing trade."""
        # Map close → entry label
        mapping = {
            'SELL_CLOSE_LONG': 'BUY_OPEN_LONG',
            'BUY_CLOSE_SHORT': 'SELL_OPEN_SHORT',
            'STOP_LONG': 'BUY_OPEN_LONG',
            'STOP_SHORT': 'SELL_OPEN_SHORT',
            'PRUNE_DEVIANCE': None, 'PRUNE_OLDEST': None,
            'PRUNE_GAP': None, 'PRUNE_FUNDING': None,
            'PRUNE_OFFSET': None,
        }
        entry_label = mapping.get(close_label)
        if entry_label and entry_label in self._open_entries:
            entries = self._open_entries[entry_label]
            if entries:
                entry = entries.pop(0)  # FIFO
                hold_ms = close_ts - entry['timestamp']
                return hold_ms / 3600000.0  # ms to hours
        return None

    def _calc_r_multiple(self, pnl: float, price: float, qty: float) -> float:
        """Calculate R-multiple (PnL / risk per unit)."""
        notional = price * qty if qty > 0 else 1.0
        if notional < 1e-9:
            return 0.0
        return pnl / (notional * 0.01)  # Risk = 1% of notional as baseline

    def _update_running_stats(self, pnl: float, hold_hours: Optional[float]):
        """Update running aggregates."""
        if abs(pnl) < 1e-12:
            return
        self._trade_pnls.append(pnl)
        self._total_trades += 1
        if pnl > 0:
            self._wins += 1
            self._gross_profit += pnl
        else:
            self._losses += 1
            self._gross_loss += abs(pnl)
        if hold_hours is not None:
            self._hold_hours.append(hold_hours)


def _ms_to_str(ms: int) -> str:
    """Convert milliseconds timestamp to formatted string."""
    if ms <= 0:
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    try:
        return datetime.fromtimestamp(
            ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    except (OSError, ValueError):
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
