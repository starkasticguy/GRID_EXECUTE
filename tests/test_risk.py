"""
Tests for risk management: Liquidation, VaR, Margin Tiers.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from core.risk import (
    get_margin_tier, calculate_maintenance_margin,
    calculate_liquidation_price, calculate_var_95,
    check_var_constraint, calculate_unrealized_pnl,
    calculate_funding_pnl,
)


class TestMarginTiers:
    def test_lowest_tier(self):
        mmr, cum = get_margin_tier(10_000)
        assert mmr == 0.004
        assert cum == 0.0

    def test_mid_tier(self):
        mmr, cum = get_margin_tier(500_000)
        assert mmr == 0.01
        assert cum == 1300.0

    def test_highest_tier(self):
        mmr, cum = get_margin_tier(200_000_000)
        assert mmr == 0.125

    def test_maintenance_margin(self):
        mm = calculate_maintenance_margin(1.0, 40000)  # 1 BTC @ 40k
        assert mm > 0


class TestLiquidation:
    def test_long_liquidation_below_entry(self):
        """Long liquidation price should be below entry."""
        liq = calculate_liquidation_price(
            entry_price=50000, position_size=0.1,
            wallet_balance=1000, side=1, leverage=5.0)
        assert 0 < liq < 50000

    def test_short_liquidation_above_entry(self):
        """Short liquidation price should be above entry."""
        liq = calculate_liquidation_price(
            entry_price=50000, position_size=0.1,
            wallet_balance=1000, side=-1, leverage=5.0)
        assert liq > 50000

    def test_no_position_no_liquidation(self):
        """Zero position should have no liquidation price."""
        liq = calculate_liquidation_price(
            entry_price=50000, position_size=0,
            wallet_balance=1000, side=1)
        assert liq == 0.0


class TestVaR:
    def test_var_positive(self):
        var = calculate_var_95(10000, 0.01)
        assert var > 0

    def test_var_scales_with_portfolio(self):
        var1 = calculate_var_95(10000, 0.01)
        var2 = calculate_var_95(20000, 0.01)
        assert abs(var2 - 2 * var1) < 1e-6

    def test_var_constraint_blocks(self):
        """VaR constraint should block when exposure is too high."""
        # Very high exposure, very small equity
        blocked = check_var_constraint(100000, 0.05, 0.15, 1000)
        assert blocked

    def test_var_constraint_allows(self):
        """VaR constraint should allow when exposure is small."""
        blocked = check_var_constraint(100, 0.01, 0.15, 10000)
        assert not blocked


class TestUnrealizedPnl:
    def test_long_profit(self):
        pnl = calculate_unrealized_pnl(100, 110, 1.0, 1)
        assert pnl == 10.0

    def test_long_loss(self):
        pnl = calculate_unrealized_pnl(100, 90, 1.0, 1)
        assert pnl == -10.0

    def test_short_profit(self):
        pnl = calculate_unrealized_pnl(100, 90, 1.0, -1)
        assert pnl == 10.0

    def test_short_loss(self):
        pnl = calculate_unrealized_pnl(100, 110, 1.0, -1)
        assert pnl == -10.0


class TestFundingPnl:
    def test_long_pays_positive_funding(self):
        pnl = calculate_funding_pnl(1.0, 50000, 0.0001, 1)
        assert pnl < 0  # Long pays

    def test_short_receives_positive_funding(self):
        pnl = calculate_funding_pnl(1.0, 50000, 0.0001, -1)
        assert pnl > 0  # Short receives


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
