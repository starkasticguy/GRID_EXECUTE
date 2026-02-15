"""
Risk Management: Liquidation, VaR, Margin Tiers, Funding PnL.

Tiered margin system per Binance USDⓈ-M specification.
Pure NumPy implementation.
"""
import numpy as np

# ─── Binance USDⓈ-M Margin Tiers (BTC example) ──────────────
# (max_notional, mmr, cum_maintenance)
# Simplified 6-tier model — covers 99%+ of backtest scenarios
MARGIN_TIERS = [
    (50_000,      0.004,  0.0),
    (250_000,     0.005,  50.0),
    (1_000_000,   0.01,   1300.0),
    (5_000_000,   0.025,  8800.0),
    (25_000_000,  0.05,   46300.0),
    (100_000_000, 0.10,   171300.0),
    (float('inf'),0.125,  296300.0),
]


def get_margin_tier(notional: float) -> tuple:
    """
    Look up MMR and cumulative maintenance for a given notional value.

    Returns:
        (mmr, cum_maintenance)
    """
    for max_notional, mmr, cum in MARGIN_TIERS:
        if notional <= max_notional:
            return mmr, cum
    # Fallback to highest tier
    return MARGIN_TIERS[-1][1], MARGIN_TIERS[-1][2]


def calculate_maintenance_margin(position_size: float, price: float) -> float:
    """
    Maintenance Margin = Notional * MMR - Cumulative

    Tier-aware calculation.
    """
    notional = abs(position_size) * price
    mmr, cum = get_margin_tier(notional)
    return notional * mmr - cum


def calculate_liquidation_price(entry_price: float, position_size: float,
                                wallet_balance: float, side: int,
                                leverage: float = 1.0,
                                max_iterations: int = 10) -> float:
    """
    Iterative liquidation price solver for USDⓈ-M linear contracts.

    For Long:
      WB + Q*(P_liq - Entry) = Q * P_liq * MMR - Cum
      → P_liq = (WB - Q*Entry + Cum) / (Q * (MMR - 1))

    For Short:
      WB + Q*(Entry - P_liq) = Q * P_liq * MMR - Cum
      → P_liq = (WB + Q*Entry + Cum) / (Q * (MMR + 1))

    Since MMR depends on Notional (= P_liq * Q), and P_liq is what
    we're solving for, we iterate until convergence.

    Args:
        side: 1 = Long, -1 = Short
    Returns:
        Liquidation price (0.0 if no position)
    """
    q = abs(position_size)
    if q < 1e-12:
        return 0.0

    # Initial guess
    p_liq = entry_price

    for _ in range(max_iterations):
        notional = p_liq * q
        mmr, cum = get_margin_tier(notional)

        if side == 1:  # Long
            denom = q * (mmr - 1.0)
            if abs(denom) < 1e-12:
                return 0.0
            p_new = (wallet_balance - q * entry_price + cum) / denom
        else:  # Short
            denom = q * (mmr + 1.0)
            if abs(denom) < 1e-12:
                return 0.0
            p_new = (wallet_balance + q * entry_price + cum) / denom

        # Convergence check
        if abs(p_new - p_liq) < 1e-6:
            break
        p_liq = p_new

    return max(0.0, p_liq)


def calculate_var_95(portfolio_value: float, sigma_15m: float) -> float:
    """
    95% Value at Risk (1-tailed, 1-period).

    VaR_95 = Portfolio × 1.65 × σ_15m
    """
    return portfolio_value * 1.65 * sigma_15m


def check_var_constraint(portfolio_value: float, sigma_15m: float,
                         max_drawdown_pct: float, equity: float) -> bool:
    """
    VaR hard cap: reject new orders if VaR > max_drawdown * equity.

    Returns True if constraint is violated (should BLOCK new orders).
    """
    var = calculate_var_95(portfolio_value, sigma_15m)
    limit = max_drawdown_pct * equity
    return var > limit


def calculate_unrealized_pnl(entry_price: float, current_price: float,
                             position_size: float, side: int) -> float:
    """
    Unrealized PnL for a position.

    Long:  (current - entry) * size
    Short: (entry - current) * size
    """
    if side == 1:
        return (current_price - entry_price) * abs(position_size)
    else:
        return (entry_price - current_price) * abs(position_size)


def calculate_funding_pnl(position_size: float, mark_price: float,
                          funding_rate: float, side: int) -> float:
    """
    Funding fee PnL per interval.

    Fee = |Position| * Price * Rate
    Long pays positive rate (negative PnL).
    Short receives positive rate (positive PnL).
    """
    notional = abs(position_size) * mark_price
    fee = notional * funding_rate
    if side == 1:
        return -fee  # Long pays
    else:
        return fee   # Short receives
