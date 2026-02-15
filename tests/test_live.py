"""
Tests for the live execution module.

Tests cover:
  - StateManager: save/load/corruption recovery
  - TradeLogger: running metrics computation
  - HealthMonitor: heartbeat, equity checks, error tracking
  - BinanceExecutor: dry-run mode
  - LiveRunner: state serialization round-trip
"""
import os
import json
import time
import shutil
import tempfile
import pytest
import numpy as np

from config import STRATEGY_PARAMS, LIVE_CONFIG
from engine.types import PositionTracker
from live.state import StateManager
from live.logger import TradeLogger
from live.monitor import HealthMonitor
from live.executor import BinanceExecutor


# ─── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def state_mgr(tmp_dir):
    return StateManager(tmp_dir, 'BTCUSDT')


@pytest.fixture
def trade_logger(tmp_dir):
    return TradeLogger(tmp_dir, 'BTCUSDT', 'WARNING')


@pytest.fixture
def health_monitor():
    return HealthMonitor(LIVE_CONFIG, initial_equity=10000.0)


# ─── StateManager Tests ─────────────────────────────────────────

class TestStateManager:

    def test_save_and_load_round_trip(self, state_mgr):
        """State saved and loaded back should match."""
        state = {
            'version': 'v4.0',
            'symbol': 'BTCUSDT',
            'wallet_balance': 10234.56,
            'halted': False,
            'grid_anchor_long': 43150.0,
            'pos_long': {
                'size': 0.046,
                'avg_entry': 43100.0,
                'fills': [
                    {'price': 43000.0, 'qty': 0.023,
                     'timestamp': 1705330800000, 'funding_cost': 5.67}
                ],
            },
            'metrics': {'longs_opened': 10},
        }
        state_mgr.save(state)
        loaded = state_mgr.load()

        assert loaded is not None
        assert loaded['wallet_balance'] == 10234.56
        assert loaded['halted'] is False
        assert loaded['pos_long']['size'] == 0.046
        assert len(loaded['pos_long']['fills']) == 1
        assert loaded['pos_long']['fills'][0]['price'] == 43000.0
        assert loaded['metrics']['longs_opened'] == 10

    def test_load_missing_file(self, state_mgr):
        """Loading when no file exists returns None."""
        result = state_mgr.load()
        assert result is None

    def test_load_corrupt_file(self, state_mgr):
        """Corrupted JSON file handled gracefully."""
        # Write invalid JSON
        with open(state_mgr.state_file, 'w') as f:
            f.write('{invalid json content')

        result = state_mgr.load()
        assert result is None

        # Original file should be renamed to .corrupt.*
        assert not os.path.exists(state_mgr.state_file)

    def test_save_trade_csv_and_jsonl(self, state_mgr):
        """Trade saved to both CSV and JSONL."""
        trade = {
            'timestamp': 1705334400000,
            'datetime': '2024-01-15T14:00:00Z',
            'symbol': 'BTCUSDT',
            'price': 43250.0,
            'qty': 0.023,
            'label': 'BUY_OPEN_LONG',
            'regime': 0,
            'pnl': 0.0,
            'exchange_order_id': '12345',
            'fill_price': 43250.0,
            'fee': 1.99,
            'wallet_balance_after': 10098.01,
            'equity_after': 10178.01,
            'pos_long_size': 0.023,
            'pos_short_size': 0.0,
        }
        state_mgr.save_trade(trade)

        # Check CSV exists and has data
        assert os.path.exists(state_mgr.trade_csv)

        # Check JSONL
        trades = state_mgr.load_trades()
        assert len(trades) == 1
        assert trades[0]['price'] == 43250.0
        assert trades[0]['label'] == 'BUY_OPEN_LONG'

    def test_multiple_trades_appended(self, state_mgr):
        """Multiple trades appended correctly."""
        for i in range(5):
            state_mgr.save_trade({
                'timestamp': 1705334400000 + i * 900000,
                'price': 43000 + i * 100,
                'label': 'BUY_OPEN_LONG',
                'qty': 0.01,
            })

        trades = state_mgr.load_trades()
        assert len(trades) == 5
        assert trades[0]['price'] == 43000
        assert trades[4]['price'] == 43400

    def test_snapshot_creation(self, state_mgr):
        """Snapshots are created with correct naming."""
        state = {'test': True, 'wallet_balance': 5000}
        state_mgr.save_snapshot(state, label='test')

        import glob
        snaps = glob.glob(os.path.join(state_mgr.state_dir, '*snapshot*'))
        assert len(snaps) == 1
        assert 'test' in snaps[0]

    def test_cleanup_old_snapshots(self, state_mgr):
        """Old snapshots are cleaned up correctly."""
        # Create 15 snapshots
        for i in range(15):
            state_mgr.save_snapshot({'n': i}, label=f'snap{i:02d}')
            time.sleep(0.01)

        import glob
        snaps = glob.glob(os.path.join(state_mgr.state_dir, '*snapshot*'))
        assert len(snaps) == 15

        state_mgr.cleanup_old_snapshots(keep_last=5)

        snaps = glob.glob(os.path.join(state_mgr.state_dir, '*snapshot*'))
        assert len(snaps) == 5


# ─── TradeLogger Tests ──────────────────────────────────────────

class TestTradeLogger:

    def test_running_metrics_empty(self, trade_logger):
        """Empty metrics when no trades logged."""
        m = trade_logger.get_running_metrics()
        assert m['total_trades'] == 0
        assert m['win_rate_pct'] == 0.0
        assert m['profit_factor'] == 0.0

    def test_running_metrics_with_trades(self, trade_logger):
        """Running metrics computed correctly from trade history."""
        # Log some winning exits
        for pnl in [10, 20, 15, -5, -8, 25, -3]:
            trade_logger._update_running_stats(pnl, 2.0)

        m = trade_logger.get_running_metrics()
        assert m['total_trades'] == 7
        assert m['win_rate_pct'] == pytest.approx(57.1, abs=0.1)
        # gross_profit = 10+20+15+25 = 70
        # gross_loss = 5+8+3 = 16
        assert m['profit_factor'] == pytest.approx(70 / 16, abs=0.01)
        assert m['total_pnl'] == pytest.approx(54, abs=0.1)
        assert m['avg_hold_hours'] == pytest.approx(2.0, abs=0.1)

    def test_win_rate_all_wins(self, trade_logger):
        """Win rate is 100% when all trades are winners."""
        for _ in range(10):
            trade_logger._update_running_stats(5.0, 1.0)

        m = trade_logger.get_running_metrics()
        assert m['win_rate_pct'] == 100.0

    def test_win_rate_all_losses(self, trade_logger):
        """Win rate is 0% when all trades are losers."""
        for _ in range(10):
            trade_logger._update_running_stats(-5.0, 1.0)

        m = trade_logger.get_running_metrics()
        assert m['win_rate_pct'] == 0.0


# ─── HealthMonitor Tests ────────────────────────────────────────

class TestHealthMonitor:

    def test_heartbeat(self, health_monitor):
        """Heartbeat updates last_heartbeat and resets errors."""
        health_monitor.consecutive_errors = 3
        health_monitor.heartbeat()

        assert health_monitor.consecutive_errors == 0
        assert health_monitor.check_heartbeat()

    def test_stale_heartbeat(self, health_monitor):
        """Stale heartbeat detected correctly."""
        health_monitor.last_heartbeat = time.time() - 2000
        assert not health_monitor.check_heartbeat(max_gap_seconds=1200)

    def test_equity_ok(self, health_monitor):
        """Equity within normal bounds."""
        assert health_monitor.check_equity(10500.0) == 'ok'

    def test_equity_warning(self, health_monitor):
        """Equity warning when below 90% of initial."""
        assert health_monitor.check_equity(8500.0) == 'warning'

    def test_equity_critical(self, health_monitor):
        """Equity critical when drawdown exceeds emergency threshold."""
        # Peak = 10000, emergency = 25%
        # 10000 * (1 - 0.25) = 7500
        assert health_monitor.check_equity(7400.0) == 'critical'

    def test_error_tracking(self, health_monitor):
        """Error tracking increments correctly."""
        for i in range(4):
            should_stop = health_monitor.report_error(RuntimeError(f"test {i}"))
            assert not should_stop

        # 5th error should trigger shutdown recommendation
        should_stop = health_monitor.report_error(RuntimeError("final"))
        assert should_stop
        assert health_monitor.consecutive_errors == 5
        assert health_monitor.error_count == 5

    def test_heartbeat_resets_consecutive(self, health_monitor):
        """Heartbeat resets consecutive error count."""
        health_monitor.report_error(RuntimeError("err1"))
        health_monitor.report_error(RuntimeError("err2"))
        assert health_monitor.consecutive_errors == 2

        health_monitor.heartbeat()
        assert health_monitor.consecutive_errors == 0
        assert health_monitor.error_count == 2  # Total still tracked

    def test_get_status(self, health_monitor):
        """Status dict returned correctly."""
        status = health_monitor.get_status()
        assert status['status'] == 'healthy'
        assert status['error_count'] == 0
        assert 'heartbeat_age_seconds' in status

    def test_peak_equity_tracking(self, health_monitor):
        """Peak equity updates on new highs."""
        health_monitor.check_equity(11000.0)
        assert health_monitor.peak_equity == 11000.0

        health_monitor.check_equity(10500.0)
        assert health_monitor.peak_equity == 11000.0  # No change

    def test_position_sync_check(self, health_monitor):
        """Position sync with matching positions returns True."""
        # Create mock executor that returns matching positions
        class MockExecutor:
            def get_positions(self, symbol):
                return {
                    'long': {'size': 0.05, 'avg_entry': 43000},
                    'short': {'size': 0.03, 'avg_entry': 44000},
                }

        pos_long = PositionTracker(side=1)
        pos_long.add_fill(43000, 0.05, time.time() * 1000, 0)
        pos_short = PositionTracker(side=-1)
        pos_short.add_fill(44000, 0.03, time.time() * 1000, 0)

        assert health_monitor.check_position_sync(
            MockExecutor(), 'BTC/USDT:USDT', pos_long, pos_short)

    def test_position_sync_mismatch(self, health_monitor):
        """Position sync with mismatched positions returns False."""
        class MockExecutor:
            def get_positions(self, symbol):
                return {
                    'long': {'size': 0.10, 'avg_entry': 43000},
                    'short': {'size': 0.0, 'avg_entry': 0},
                }

        pos_long = PositionTracker(side=1)
        pos_long.add_fill(43000, 0.05, time.time() * 1000, 0)
        pos_short = PositionTracker(side=-1)

        assert not health_monitor.check_position_sync(
            MockExecutor(), 'BTC/USDT:USDT', pos_long, pos_short)


# ─── BinanceExecutor Dry-Run Tests ──────────────────────────────

class TestBinanceExecutorDryRun:

    @pytest.fixture
    def dry_executor(self):
        """Create a dry-run executor (no real exchange connection needed)."""
        config = LIVE_CONFIG.copy()
        config['max_retry_attempts'] = 1
        return BinanceExecutor(
            'test_key', 'test_secret', config,
            dry_run=True, testnet=False)

    def test_dry_run_limit_order(self, dry_executor):
        """Dry-run limit order returns synthetic order dict."""
        order = dry_executor.place_limit_order(
            'BTC/USDT:USDT', 'buy', 0.01, 43000.0, 'LONG')

        assert order is not None
        assert order['id'].startswith('DRY_')
        assert order['side'] == 'buy'
        assert order['amount'] == 0.01
        assert order['price'] == 43000.0
        assert order['type'] == 'limit'

    def test_dry_run_market_order(self, dry_executor):
        """Dry-run market order returns synthetic order dict."""
        order = dry_executor.place_market_order(
            'BTC/USDT:USDT', 'sell', 0.01, 'LONG', reduce_only=True)

        assert order is not None
        assert order['id'].startswith('DRY_')
        assert order['side'] == 'sell'
        assert order['type'] == 'market'

    def test_dry_run_cancel_all(self, dry_executor):
        """Dry-run cancel returns 0."""
        result = dry_executor.cancel_all_orders('BTC/USDT:USDT')
        assert result == 0

    def test_dry_run_cancel_order(self, dry_executor):
        """Dry-run cancel specific order returns True."""
        result = dry_executor.cancel_order('12345', 'BTC/USDT:USDT')
        assert result is True

    def test_dry_run_unique_ids(self, dry_executor):
        """Each dry-run order gets a unique ID."""
        ids = set()
        for _ in range(10):
            order = dry_executor.place_limit_order(
                'BTC/USDT:USDT', 'buy', 0.01, 42000.0, 'LONG')
            ids.add(order['id'])

        assert len(ids) == 10

    def test_dry_run_reduce_only_flagged(self, dry_executor):
        """Dry-run order includes reduceOnly in info."""
        order = dry_executor.place_limit_order(
            'BTC/USDT:USDT', 'sell', 0.01, 44000.0, 'LONG',
            reduce_only=True)
        assert order['info']['reduceOnly'] == 'true'


# ─── State Serialization Round-Trip ──────────────────────────────

class TestStateRoundTrip:

    def test_position_tracker_serialization(self):
        """PositionTracker state survives serialize → save → load → restore."""
        # Build a position with multiple fills
        pos = PositionTracker(side=1)
        pos.add_fill(43000, 0.02, 1705330800000, 1.0)
        pos.add_fill(42800, 0.03, 1705331700000, 1.5)
        pos.add_funding(-5.0)

        assert pos.size == pytest.approx(0.05)
        assert pos.num_fills == 2
        assert pos.funding_pnl == pytest.approx(-5.0)

        # Serialize
        fills_data = [
            {'price': f['price'], 'qty': f['qty'],
             'timestamp': f['timestamp'], 'funding_cost': f['funding_cost']}
            for f in pos.fills
        ]

        # Restore into a new tracker
        pos2 = PositionTracker(side=1)
        for f in fills_data:
            pos2.add_fill(f['price'], f['qty'], f['timestamp'], 0)
            pos2.fills[-1]['funding_cost'] = f['funding_cost']

        assert pos2.size == pytest.approx(pos.size, abs=1e-9)
        assert pos2.num_fills == pos.num_fills
        assert pos2.avg_entry == pytest.approx(pos.avg_entry, abs=0.01)
        assert len(pos2.fills) == 2
        assert pos2.fills[0]['funding_cost'] == pytest.approx(2.5, abs=0.01)

    def test_full_state_dict_serialization(self, tmp_dir):
        """Full state dict survives JSON round-trip."""
        state = {
            'version': 'v4.0',
            'symbol': 'BTCUSDT',
            'timestamp': int(time.time() * 1000),
            'wallet_balance': 10500.25,
            'initial_capital': 10000.0,
            'pos_long': {
                'size': 0.05,
                'avg_entry': 43000.0,
                'realized_pnl': 200.0,
                'unrealized_pnl': 50.0,
                'funding_pnl': -10.0,
                'num_fills': 2,
                'fills': [
                    {'price': 43100, 'qty': 0.02, 'timestamp': 1e12, 'funding_cost': 3.0},
                    {'price': 42900, 'qty': 0.03, 'timestamp': 1e12 + 1000, 'funding_cost': 7.0},
                ],
            },
            'pos_short': {
                'size': 0.0,
                'avg_entry': 0.0,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'funding_pnl': 0.0,
                'num_fills': 0,
                'fills': [],
            },
            'halted': False,
            'grid_anchor_long': 43050.0,
            'grid_anchor_short': 43500.0,
            'trailing_anchor': 43050.0,
            'grid_needs_regen': True,
            'accumulated_profit_long': 150.0,
            'accumulated_profit_short': 0.0,
            'bar_count': 100,
            'last_processed_bar_ts': int(time.time() * 1000),
            'active_orders': {
                'order_1': {
                    'side': 'buy', 'price': 42800, 'qty': 0.02,
                    'direction': 1, 'reduce_only': False,
                    'grid_level': 0, 'position_side': 'LONG',
                    'placed_at': int(time.time() * 1000),
                },
            },
            'metrics': {
                'longs_opened': 10, 'longs_closed': 8,
                'shorts_opened': 5, 'shorts_closed': 4,
                'stops_long': 1, 'stops_short': 0,
                'prune_count': 2, 'prune_types': {'PRUNE_OLDEST': 2},
                'circuit_breaker_triggers': 0,
                'trailing_shifts': 3,
                'funding_pnl': -10.0,
                'var_blocks': 1,
                'liquidations': 0,
            },
        }

        mgr = StateManager(tmp_dir, 'BTCUSDT')
        mgr.save(state)

        loaded = mgr.load()
        assert loaded is not None

        # Verify key fields
        assert loaded['wallet_balance'] == 10500.25
        assert loaded['pos_long']['size'] == 0.05
        assert len(loaded['pos_long']['fills']) == 2
        assert loaded['active_orders']['order_1']['side'] == 'buy'
        assert loaded['metrics']['longs_opened'] == 10
        assert loaded['metrics']['prune_types']['PRUNE_OLDEST'] == 2


# ─── Indicator Buffer Test ───────────────────────────────────────

class TestIndicatorBuffer:

    def test_indicators_on_buffer_match_full(self):
        """Indicators computed on 200-bar buffer match those from full dataset."""
        from core.kama import calculate_er, calculate_kama, detect_regime
        from core.atr import calculate_atr, calculate_z_score, calculate_rolling_volatility

        # Generate synthetic price data (300 bars)
        np.random.seed(42)
        n = 300
        base_price = 43000
        changes = np.random.randn(n) * 100
        closes = base_price + np.cumsum(changes)
        highs = closes + np.abs(np.random.randn(n) * 50)
        lows = closes - np.abs(np.random.randn(n) * 50)

        config = STRATEGY_PARAMS

        # Compute on full dataset
        er_full = calculate_er(closes, config['kama_period'])
        kama_full = calculate_kama(closes, er_full,
                                    config['kama_fast'], config['kama_slow'])
        atr_full = calculate_atr(highs, lows, closes, config['atr_period'])
        regime_full = detect_regime(kama_full, er_full, atr_full,
                                     config['regime_threshold'],
                                     config['er_trend_threshold'])
        z_full = calculate_z_score(closes, atr_full, 20)
        vol_full = calculate_rolling_volatility(closes, config['atr_period'])

        # Compute on last 200 bars only (buffer)
        buf_start = 100
        c_buf = closes[buf_start:]
        h_buf = highs[buf_start:]
        l_buf = lows[buf_start:]

        er_buf = calculate_er(c_buf, config['kama_period'])
        kama_buf = calculate_kama(c_buf, er_buf,
                                   config['kama_fast'], config['kama_slow'])
        atr_buf = calculate_atr(h_buf, l_buf, c_buf, config['atr_period'])
        regime_buf = detect_regime(kama_buf, er_buf, atr_buf,
                                    config['regime_threshold'],
                                    config['er_trend_threshold'])

        # The last N values should converge (KAMA is recursive, so early
        # values differ but later ones should be close)
        # Check last 50 bars where indicators are fully warmed
        assert np.allclose(
            atr_full[-50:], atr_buf[-50:], rtol=0.01), \
            "ATR should match within 1% for warmed bars"

        # Regime should match for recent bars
        assert np.array_equal(
            regime_full[-20:], regime_buf[-20:]), \
            "Regime should match for recent bars"
