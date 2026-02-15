"""
HealthMonitor — System health checks for live trading.

Checks:
  - Heartbeat (is the loop still running?)
  - Position sync (internal tracker matches exchange)
  - Balance monitoring (equity within bounds)
  - Error rate tracking
"""
import time
import logging

from engine.types import PositionTracker

logger = logging.getLogger('health_monitor')


class HealthMonitor:

    def __init__(self, config: dict, initial_equity: float):
        self.config = config
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity

        self.last_heartbeat = time.time()
        self.error_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.emergency_stop_loss_pct = config.get('emergency_stop_loss_pct', 0.25)
        self.sync_tolerance = config.get('position_sync_tolerance_pct', 0.02)

    def heartbeat(self):
        """Record a successful loop iteration."""
        self.last_heartbeat = time.time()
        self.consecutive_errors = 0

    def check_heartbeat(self, max_gap_seconds: float = 1200) -> bool:
        """Check if heartbeat is recent (within 20 min). Returns True if healthy."""
        gap = time.time() - self.last_heartbeat
        if gap > max_gap_seconds:
            logger.warning(f"Heartbeat stale: {gap:.0f}s since last beat "
                           f"(max {max_gap_seconds}s)")
            return False
        return True

    def check_position_sync(self, executor, symbol: str,
                             pos_long: PositionTracker,
                             pos_short: PositionTracker) -> bool:
        """Verify internal position tracker matches exchange. Returns True if synced."""
        try:
            exchange_pos = executor.get_positions(symbol)
        except Exception as e:
            logger.warning(f"Position sync check failed: {e}")
            return True  # Assume OK if we can't check

        ex_long = exchange_pos['long']['size']
        ex_short = exchange_pos['short']['size']
        int_long = pos_long.size
        int_short = pos_short.size

        # Check long side
        long_diff = abs(ex_long - int_long)
        long_base = max(ex_long, int_long, 1e-9)
        long_pct = long_diff / long_base

        # Check short side
        short_diff = abs(ex_short - int_short)
        short_base = max(ex_short, int_short, 1e-9)
        short_pct = short_diff / short_base

        synced = True
        if long_pct > self.sync_tolerance and long_diff > 1e-8:
            logger.warning(
                f"LONG position mismatch: exchange={ex_long:.8f} "
                f"internal={int_long:.8f} (diff={long_pct:.2%})")
            synced = False

        if short_pct > self.sync_tolerance and short_diff > 1e-8:
            logger.warning(
                f"SHORT position mismatch: exchange={ex_short:.8f} "
                f"internal={int_short:.8f} (diff={short_pct:.2%})")
            synced = False

        return synced

    def check_equity(self, current_equity: float) -> str:
        """
        Check current equity against thresholds.

        Returns:
          'ok'       — equity within normal bounds
          'warning'  — equity below 90% of initial
          'critical' — equity below emergency stop loss threshold
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / max(self.peak_equity, 1e-9)
        initial_loss = (self.initial_equity - current_equity) / max(self.initial_equity, 1e-9)

        if drawdown >= self.emergency_stop_loss_pct:
            logger.critical(
                f"EMERGENCY: Drawdown {drawdown:.2%} exceeds "
                f"limit {self.emergency_stop_loss_pct:.2%}! "
                f"Equity: ${current_equity:,.2f}")
            return 'critical'

        if initial_loss > 0.10:
            logger.warning(
                f"Equity warning: down {initial_loss:.2%} from initial "
                f"(${current_equity:,.2f} vs ${self.initial_equity:,.2f})")
            return 'warning'

        return 'ok'

    def report_error(self, error: Exception) -> bool:
        """
        Track an error. Returns True if consecutive errors exceed threshold
        (caller should shut down).
        """
        self.error_count += 1
        self.consecutive_errors += 1
        logger.error(f"Error #{self.error_count} "
                     f"(consecutive: {self.consecutive_errors}): {error}")

        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.critical(
                f"Consecutive errors ({self.consecutive_errors}) exceeded "
                f"max ({self.max_consecutive_errors}). Recommending shutdown.")
            return True
        return False

    def get_status(self) -> dict:
        """Return current health status summary."""
        gap = time.time() - self.last_heartbeat
        if self.consecutive_errors >= self.max_consecutive_errors:
            status = 'critical'
        elif self.consecutive_errors > 0 or gap > 600:
            status = 'degraded'
        else:
            status = 'healthy'

        return {
            'last_heartbeat': self.last_heartbeat,
            'heartbeat_age_seconds': round(gap, 1),
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'status': status,
        }
