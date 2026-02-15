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
        """Fill far from current price should be pruned."""
        pos = make_long_position([(80, 1.0, 0), (100, 1.0, 100)])
        # Price at 120, grid_spacing=5, gap_mult=3 → threshold=15
        # Fill at 80 is 40 away > 15 → prune
        idx = check_gap_prune(pos, 120.0, 100.0, 5.0, 3.0)
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
        """Should prune worst loser when profit buffer is available."""
        pos = make_long_position([(120, 1.0, 0), (100, 1.0, 100)])
        # Price at 95 → fill at 120 is losing (120-95)*1 = 25
        # accumulated_profit = 20 > 25*0.5=12.5 → prune
        idx = check_profit_offset_prune(pos, 95.0, accumulated_profit=20.0)
        assert idx == 0  # Fill at 120 is worst

    def test_no_profit_buffer(self):
        """Should not prune without sufficient profit."""
        pos = make_long_position([(120, 1.0, 0)])
        idx = check_profit_offset_prune(pos, 95.0, accumulated_profit=0.0)
        assert idx == -1


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
