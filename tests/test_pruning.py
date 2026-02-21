"""
Tests for the 5-Method Pruning ("Gardener") module.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from engine.types import PositionTracker
from engine.pruning import (
    check_deviance_prune, check_oldest_prune, check_gap_prune,
    check_funding_prune, check_profit_offset_prune, run_pruning_cycle,
)
from core.kama import apply_regime_hysteresis, resample_ohlcv, REGIME_NOISE, REGIME_UPTREND, REGIME_DOWNTREND
import numpy as np


def make_long_position(fills):
    """Create a long position with given fills: [(price, qty, timestamp), ...]"""
    pos = PositionTracker(side=1)
    for price, qty, ts in fills:
        pos.add_fill(price, qty, ts)
    return pos


def make_short_position(fills):
    """Create a short position with given fills."""
    pos = PositionTracker(side=-1)
    for price, qty, ts in fills:
        pos.add_fill(price, qty, ts)
    return pos


class TestDeviancePrune:
    def test_long_deviance_triggers(self):
        """Long position should prune when price crashes far below KAMA."""
        pos = make_long_position([(100, 1.0, 0), (105, 1.0, 100)])
        # Price crashed to 60, KAMA at 100, ATR = 5, threshold = 3
        # 60 < 100 - 3*5 = 85 → prune!
        idx = check_deviance_prune(pos, 60.0, 100.0, 5.0, 3.0)
        assert idx >= 0

    def test_long_no_deviance(self):
        """Should NOT prune if price is near KAMA."""
        pos = make_long_position([(100, 1.0, 0)])
        idx = check_deviance_prune(pos, 98.0, 100.0, 5.0, 3.0)
        assert idx == -1

    def test_short_deviance_triggers(self):
        """Short position should prune when price pumps far above KAMA."""
        pos = make_short_position([(100, 1.0, 0)])
        # Price pumped to 130, KAMA=100, ATR=5 → 130 > 100+15=115 → prune
        idx = check_deviance_prune(pos, 130.0, 100.0, 5.0, 3.0)
        assert idx >= 0

    def test_prunes_worst_fill(self):
        """Should prune the worst fill (highest entry for long)."""
        pos = make_long_position([(90, 1.0, 0), (110, 1.0, 100), (95, 1.0, 200)])
        idx = check_deviance_prune(pos, 60.0, 100.0, 5.0, 3.0)
        assert idx == 1  # Fill at 110 is worst for longs


class TestOldestPrune:
    def test_old_fill_pruned(self):
        """Fills older than max_age should be pruned."""
        pos = make_long_position([
            (100, 1.0, 0),         # Very old
            (102, 1.0, 80000),     # Recent
        ])
        idx = check_oldest_prune(pos, 90000, max_age_seconds=86400)
        assert idx == 0  # Age = 90000 > 86400 → prune first

    def test_no_old_fills(self):
        """Nothing should prune if all fills are recent."""
        pos = make_long_position([(100, 1.0, 80000)])
        idx = check_oldest_prune(pos, 85000, max_age_seconds=86400)
        assert idx == -1


class TestGapPrune:
    def test_gap_too_large(self):
        """Underwater fill far from current price should be pruned."""
        pos = make_long_position([(120, 1.0, 0), (100, 1.0, 100)])
        # Price at 80, grid_spacing=5, gap_mult=3 → threshold=15
        # Fill at 120 is 40 away > 15 AND underwater (80 < 120) → prune
        idx = check_gap_prune(pos, 80.0, 100.0, 5.0, 3.0)
        assert idx == 0

    def test_no_gap(self):
        """Should not prune if fills are within gap threshold."""
        pos = make_long_position([(98, 1.0, 0)])
        idx = check_gap_prune(pos, 100.0, 100.0, 5.0, 3.0)
        assert idx == -1


class TestFundingPrune:
    def test_high_funding_cost(self):
        """Fill with high accumulated funding cost should prune."""
        pos = make_long_position([(100, 1.0, 0)])
        pos.fills[0]['funding_cost'] = 10.0
        idx = check_funding_prune(pos, grid_profit_potential=15.0, cost_ratio=0.5)
        # 10 > 0.5 * 15 = 7.5 → prune
        assert idx == 0

    def test_low_funding_cost(self):
        """Should not prune if funding cost is low."""
        pos = make_long_position([(100, 1.0, 0)])
        pos.fills[0]['funding_cost'] = 1.0
        idx = check_funding_prune(pos, grid_profit_potential=15.0, cost_ratio=0.5)
        assert idx == -1


class TestProfitOffsetPrune:
    def test_subsidized_close(self):
        """Should prune worst loser when profit buffer covers offset_ratio × loss."""
        pos = make_long_position([(120, 1.0, 0), (100, 1.0, 100)])
        # Price at 95 → fill at 120 losing (120-95)*1 = 25
        # With offset_ratio=1.0: profit=30 > 25*1.0=25 → prune
        idx = check_profit_offset_prune(pos, 95.0, accumulated_profit=30.0, offset_ratio=1.0)
        assert idx == 0  # Fill at 120 is worst

    def test_no_profit_buffer(self):
        """Should not prune without sufficient profit."""
        pos = make_long_position([(120, 1.0, 0)])
        idx = check_profit_offset_prune(pos, 95.0, accumulated_profit=0.0)
        assert idx == -1


class TestProfitOffsetRatio:
    def test_high_ratio_suppresses_prune(self):
        """With offset_ratio=1.5 (default), same profit that triggered at 0.5 should NOT trigger."""
        pos = make_long_position([(120, 1.0, 0)])
        # Price at 95, loss =25, accumulated_profit=20
        # Old threshold (0.5): 20 > 25*0.5=12.5 → would prune
        # New threshold (1.5): 20 > 25*1.5=37.5 → should NOT prune
        idx = check_profit_offset_prune(pos, 95.0, accumulated_profit=20.0, offset_ratio=1.5)
        assert idx == -1

    def test_high_ratio_fires_when_profit_sufficient(self):
        """With offset_ratio=1.5, fires when profit > 1.5 × loss."""
        pos = make_long_position([(120, 1.0, 0)])
        # loss=25, offset_ratio=1.5 → need profit > 37.5 → use 40
        idx = check_profit_offset_prune(pos, 95.0, accumulated_profit=40.0, offset_ratio=1.5)
        assert idx == 0


class TestPruneCooldown:
    def test_cooldown_suppresses_non_deviance_prunes(self):
        """When on cooldown, gap/oldest/offset prunes should be suppressed."""
        # Position with an underwater fill far from price (gap prune would normally fire)
        pos = make_long_position([(150, 1.0, 0)])
        config = {
            'deviance_sigma': 5.0,          # deviance won't fire
            'max_position_age_hours': 24,
            'gap_prune_mult': 2.0,           # gap would fire without cooldown
            'funding_cost_ratio': 0.5,
            'offset_prune_ratio': 1.5,
        }
        # Price at 100, fill at 150 (underwater) → gap = 50, spacing = 5, threshold = 10 → gap fires normally
        idx_no_cd, label_no_cd = run_pruning_cycle(
            pos, 100.0, 1000, 110.0, 5.0, 110.0, 5.0, 10.0, 0.0, config,
            is_on_cooldown=False)
        assert label_no_cd == 'PRUNE_GAP'

        # Same setup but on cooldown → gap suppressed
        idx_cd, label_cd = run_pruning_cycle(
            pos, 100.0, 1000, 110.0, 5.0, 110.0, 5.0, 10.0, 0.0, config,
            is_on_cooldown=True)
        assert idx_cd == -1
        assert label_cd is None

    def test_cooldown_does_not_suppress_deviance(self):
        """Deviance prune should always fire regardless of cooldown — toxic position."""
        pos = make_long_position([(100, 1.0, 0)])
        config = {
            'deviance_sigma': 3.0,
            'max_position_age_hours': 24,
            'gap_prune_mult': 3.0,
            'funding_cost_ratio': 0.5,
            'offset_prune_ratio': 1.5,
        }
        # Price crashed to 60: 60 < 100 - 3*5=85 → deviance
        idx, label = run_pruning_cycle(
            pos, 60.0, 1000, 100.0, 5.0, 100.0, 5.0, 10.0, 0.0, config,
            is_on_cooldown=True)  # On cooldown!
        assert label == 'PRUNE_DEVIANCE'  # Still fires


class TestRegimeHysteresis:
    def test_single_bar_flip_suppressed(self):
        """A single-bar regime change should not propagate with min_bars=3."""
        # NOISE...NOISE, then 1 bar UPTREND, then NOISE again
        regime = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
        smoothed = apply_regime_hysteresis(regime, min_bars=3)
        # The single UPTREND bar should be filtered out
        assert smoothed[4] == 0

    def test_sustained_regime_propagates(self):
        """A regime that persists min_bars bars should be confirmed."""
        # 3 consecutive UPTREND bars
        regime = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
        smoothed = apply_regime_hysteresis(regime, min_bars=3)
        # After 3 bars of UPTREND, should be confirmed
        assert smoothed[-1] == 1

    def test_breakout_is_immediate(self):
        """Breakout regimes (|code|==2) should take effect without delay."""
        from core.kama import REGIME_BREAKOUT_UP
        regime = np.array([0, 0, 0, 2, 0, 0], dtype=np.int8)
        smoothed = apply_regime_hysteresis(regime, min_bars=3)
        # Breakout at index 3 should be reflected immediately
        assert smoothed[3] == REGIME_BREAKOUT_UP


class TestResampling:
    def test_resample_output_length_matches_input(self):
        """Resampled arrays should have same length as input."""
        n = 100
        close = np.linspace(100, 200, n)
        high = close + 1.0
        low = close - 1.0
        htf_c, htf_h, htf_l = resample_ohlcv(close, high, low, mult=4)
        assert len(htf_c) == n
        assert len(htf_h) == n
        assert len(htf_l) == n

    def test_resample_high_is_max_of_window(self):
        """HTF high should be the max of the 4-bar window."""
        high = np.array([1.0, 5.0, 3.0, 2.0,   # HTF bar 0: max=5
                         4.0, 6.0, 2.0, 1.0,   # HTF bar 1: max=6
                         2.0, 2.0, 2.0, 2.0],  # HTF bar 2: max=2
                        dtype=np.float64)
        close = np.ones(12, dtype=np.float64) * 3.0
        low = np.ones(12, dtype=np.float64) * 1.0
        _, htf_h, _ = resample_ohlcv(close, high, low, mult=4)
        # At bar 4 (start of HTF bar 1), the last completed HTF bar (0) has max=5
        assert htf_h[4] == 5.0
        # At bar 8 (start of HTF bar 2), the last completed HTF bar (1) has max=6
        assert htf_h[8] == 6.0


class TestRunPruningCycle:
    def test_priority_order(self):
        """Deviance should trigger before oldest."""
        pos = make_long_position([
            (100, 1.0, 0),  # Old AND deviant
        ])
        config = {
            'deviance_sigma': 3.0,
            'max_position_age_hours': 24,
            'gap_prune_mult': 3.0,
            'funding_cost_ratio': 0.5,
        }
        # Price crashed to 60, KAMA=100, ATR=5 → deviance triggers first
        idx, label = run_pruning_cycle(
            pos, current_price=60.0, current_time=100000,
            kama_price=100.0, atr=5.0,
            grid_anchor=100.0, grid_spacing=5.0,
            grid_profit_potential=10.0, accumulated_profit=0.0,
            config=config)
        assert label == "PRUNE_DEVIANCE"

    def test_empty_position(self):
        """Empty position should not trigger any pruning."""
        pos = PositionTracker(side=1)
        config = {'deviance_sigma': 3.0, 'max_position_age_hours': 24,
                  'gap_prune_mult': 3.0, 'funding_cost_ratio': 0.5}
        idx, label = run_pruning_cycle(
            pos, 100, 1000, 100, 5, 100, 5, 10, 0, config)
        assert idx == -1
        assert label is None


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
