"""
Grid level generation + dynamic spacing.

Pure NumPy implementation.
"""
import numpy as np


# Round-number magnets where crypto stop-hunts cluster
# Bot avoids placing orders within ROUND_AVOID_PCT of these boundaries.
_ROUND_INCREMENTS = [10_000, 5_000, 1_000, 500, 100, 50, 10, 5, 1]
ROUND_AVOID_PCT   = 0.001   # 0.10% proximity threshold
ROUND_NUDGE_PCT   = 0.0015  # 0.15% nudge away from the magnet


def _nearest_round(price: float) -> float:
    """Return the nearest round-number magnet for the given price."""
    for inc in _ROUND_INCREMENTS:
        if price >= inc * 2:
            return round(price / inc) * inc
    return round(price)


def _avoid_round_numbers(price: float) -> float:
    """
    If price is within ROUND_AVOID_PCT of a round-number magnet, nudge it
    ROUND_NUDGE_PCT away (outward from the magnet) to avoid stop-hunt clusters.
    """
    if price <= 0:
        return price
    nearest = _nearest_round(price)
    if nearest <= 0:
        return price
    pct_diff = abs(price - nearest) / nearest
    if pct_diff < ROUND_AVOID_PCT:
        # nudge away — if price is above the magnet go up, else go down
        direction = 1.0 if price >= nearest else -1.0
        price = price + direction * nearest * ROUND_NUDGE_PCT
    return price


def generate_grid_levels(anchor_price: float,
                         buy_spacing: float,
                         sell_spacing: float,
                         grid_levels_count: int,
                         min_price: float = 0.0) -> tuple:
    """
    Generate arithmetic grid levels centered on anchor price.

    Buy levels:  anchor - spacing * (1, 2, ..., N)
    Sell levels: anchor + spacing * (1, 2, ..., N)

    Returns:
        (buy_levels, sell_levels) — both descending/ascending from anchor
    """
    buy_levels = np.zeros(grid_levels_count, dtype=np.float64)
    sell_levels = np.zeros(grid_levels_count, dtype=np.float64)

    for i in range(grid_levels_count):
        b = anchor_price - buy_spacing * (i + 1)
        buy_levels[i] = max(b, min_price)
        sell_levels[i] = anchor_price + sell_spacing * (i + 1)

    return buy_levels, sell_levels


def calculate_order_qty(capital: float, price: float, order_pct: float,
                        leverage: float = 1.0) -> float:
    """
    Order size in base asset terms.

    Qty = (Capital * OrderPct * Leverage) / Price
    """
    if price <= 1e-9:
        return 0.0
    return (capital * order_pct * leverage) / price


def calculate_dynamic_spacing(atr: float, spacing_k: float,
                              spacing_floor: float = 0.005,
                              ref_price: float = 0.0) -> float:
    """
    Dynamic grid spacing = max(k * ATR, floor * price).

    Volatility expands → grid widens → covers more range
    Volatility contracts → grid tightens → captures smaller moves
    """
    atr_spacing = spacing_k * atr
    floor_spacing = spacing_floor * ref_price if ref_price > 0 else 0.0
    return max(atr_spacing, floor_spacing)


def calculate_adaptive_floor(atr: float, price: float,
                             base_floor: float = 0.005,
                             vol_scale: float = 1.5) -> float:
    """
    Adaptive spacing floor that auto-scales with each coin's volatility.

    floor = max(base_floor, (ATR / price) * vol_scale)

    This means high-volatility coins (ETH, SOL) automatically get wider
    floors than low-volatility coins (BTC at high price) without manual
    per-coin configuration.

    Args:
        atr:        Current ATR value in price units
        price:      Current reference price
        base_floor: Absolute minimum floor as fraction of price (default 0.5%)
        vol_scale:  Multiplier on ATR/price to set the adaptive component

    Returns:
        Spacing floor in price units (already multiplied by price)

    Example with ETH ($3000, ATR=$40):
        ATR/price = 0.0133, * 1.5 = 0.020 → 2.0% floor → $60
        vs static 0.5% floor = $15
    """
    if price < 1e-9:
        return base_floor * 1000.0  # fallback

    atr_ratio = atr / price          # dimensionless vol ratio
    adaptive = atr_ratio * vol_scale  # scale by user param

    # Final floor = max(base, adaptive), then convert to price units
    floor_pct = max(base_floor, adaptive)
    return floor_pct * price


def generate_geometric_grid_levels(anchor_price: float,
                                    buy_spacing_pct: float,
                                    sell_spacing_pct: float,
                                    grid_levels_count: int,
                                    min_price: float = 0.0) -> tuple:
    """
    Generate geometric (percentage-based) grid levels with round-number avoidance.

    Buy levels:  anchor / (1 + pct)^i
    Sell levels: anchor * (1 + pct)^i

    Geometric spacing naturally widens outer levels — more appropriate for
    crypto where distant reversions are less certain. Round-number avoidance
    nudges levels away from stop-hunt magnets ($60K, $65K, etc).

    Returns:
        (buy_levels, sell_levels)
    """
    buy_levels  = np.zeros(grid_levels_count, dtype=np.float64)
    sell_levels = np.zeros(grid_levels_count, dtype=np.float64)

    buy_pct  = max(buy_spacing_pct,  1e-6)
    sell_pct = max(sell_spacing_pct, 1e-6)

    for i in range(grid_levels_count):
        b = anchor_price / ((1.0 + buy_pct)  ** (i + 1))
        s = anchor_price * ((1.0 + sell_pct) ** (i + 1))
        buy_levels[i]  = _avoid_round_numbers(max(b, min_price))
        sell_levels[i] = _avoid_round_numbers(s)

    return buy_levels, sell_levels

